#!/usr/bin/env python3
"""
smote_adasyn_baselines.py

SMOTE and ADASYN augmentation baselines for the DenseGAAL active learning pipeline.

These baselines mirror the structure of `run_augmented` in the main runner:
  1. Run active learning loops (any acquisition function / pooling method).
  2. Apply resampling augmentation (SMOTE or ADASYN) to the resulting labeled set.
  3. Train the final classifier on the augmented set and evaluate on the test set.

Usage (standalone, mirrors main runner CLI pattern):
    python smote_adasyn_baselines.py \
        --al_method base \
        --al_function margin \
        --resampler SMOTE \
        --classifier MLP \
        --dataset shuttle \
        --budget 50

Flags inherited from the main runner that are respected here:
    --al_method, --pooling_method, --al_function, --classifier,
    --dataset, --budget, --random_state, --al_steps

Resampler-specific flags (new):
    --resampler          {SMOTE, ADASYN, SMOTE+AL, ADASYN+AL}
                         SMOTE / ADASYN: apply to labeled set after AL.
                         SMOTE+AL / ADASYN+AL: pair with the best AL set-construction
                         strategy (AnchorAL) to test the strongest resampling baseline.
    --smote_k_neighbors  k for SMOTE (default: 5).
    --adasyn_n_neighbors k for ADASYN (default: 5).
    --sampling_strategy  Passed directly to imbalanced-learn.
                         'auto'      -> resample all minority classes to match majority.
                         'not majority' -> resample all but the majority class.
                         'budget'    -> cap total synthetic samples to --budget * --al_steps,
                                        distributed proportionally across minority classes
                                        by their deficit from the majority. This makes the
                                        comparison directly controlled against DenseGAAL.
                         float       -> ratio of minority to majority after resampling.
                         dict        -> {class_label: target_count} explicit targets.
                         (default: 'auto')

Integration note:
    To call from the main runner, import `run_smote_baseline` / `run_adasyn_baseline`
    and add them to the dispatch in main(), e.g.:

        elif args.al_method == 'SMOTE':
            y_pred, y_true, step_metrics = run_smote_baseline(args, res)
        elif args.al_method == 'ADASYN':
            y_pred, y_true, step_metrics = run_adasyn_baseline(args, res)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
except ImportError as exc:
    raise ImportError(
        "imbalanced-learn is required for SMOTE/ADASYN baselines.\n"
        "Install with: pip install imbalanced-learn"
    ) from exc

# ---------------------------------------------------------------------------
# Re-use helpers from the main runner (same package).
# All imports below mirror what the main runner uses so this file can be
# dropped into the same directory without changes.
# ---------------------------------------------------------------------------
from config import get_config
from active_learning_functions import METHOD_DICT
from classifiers.mlp import TorchMLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pool_filters_cumulative import subpool_anchoral, subpool_randsub, subpool_seals
import torch


# ===========================================================================
# 1.  Resampler classes
# ===========================================================================

def compute_budget_capped_strategy(
    y: np.ndarray,
    total_budget: int,
) -> Dict[Any, int]:
    """
    Build an imbalanced-learn `sampling_strategy` dict that caps the total
    number of synthetic samples to `total_budget`, distributed across
    minority classes proportionally to their deficit from the majority class.

    This makes SMOTE/ADASYN directly comparable to DenseGAAL, which also
    operates under a fixed synthetic sample budget.

    Parameters
    ----------
    y : np.ndarray
        Label vector of the current labeled set (after AL).
    total_budget : int
        Maximum total synthetic samples to generate across all classes.
        Typically set to --budget * --al_steps to match the AL annotation budget.

    Returns
    -------
    strategy : dict {class_label: target_count}
        Passed directly to imblearn as sampling_strategy.
        Only includes classes that need upsampling (deficit > 0).
        Classes already at or above majority count are excluded.

    Example
    -------
    If majority class has 500 samples and two minority classes have 50 and 100,
    their deficits are 450 and 400. Total deficit = 850. With budget=100:
      - class A gets round(100 * 450/850) = 53 synthetic -> target 50+53 = 103
      - class B gets round(100 * 400/850) = 47 synthetic -> target 100+47 = 147
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(classes, counts))
    maj_count = int(counts.max())

    # Compute per-class deficit (how many samples short of majority)
    deficits = {cls: max(0, maj_count - count_map[cls]) for cls in classes}

    # Only consider classes that are actually underrepresented
    minority_classes = {cls: d for cls, d in deficits.items() if d > 0}

    if not minority_classes:
        warnings.warn(
            "compute_budget_capped_strategy: no minority classes found "
            "(dataset may already be balanced). Returning empty strategy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    total_deficit = sum(minority_classes.values())

    # Distribute budget proportionally to deficit
    raw_allocs = {
        cls: total_budget * (deficit / total_deficit)
        for cls, deficit in minority_classes.items()
    }

    # Floor allocations and distribute remainder to largest fractional parts
    base_allocs = {cls: int(v) for cls, v in raw_allocs.items()}
    remainder = total_budget - sum(base_allocs.values())
    frac_order = sorted(
        minority_classes.keys(),
        key=lambda c: -(raw_allocs[c] - base_allocs[c])
    )
    for i, cls in enumerate(frac_order):
        if i >= remainder:
            break
        base_allocs[cls] += 1

    # Build target counts: current count + allocated synthetic samples.
    # Cap each class at majority count to avoid overshooting.
    strategy = {}
    for cls, n_synthetic in base_allocs.items():
        if n_synthetic <= 0:
            continue
        target = min(count_map[cls] + n_synthetic, maj_count)
        # imbalanced-learn requires target strictly greater than current count
        if target > count_map[cls]:
            strategy[cls] = int(target)

    # Log what we computed
    print(f"[budget-capped strategy] total_budget={total_budget}")
    for cls, target in strategy.items():
        n_syn = target - count_map[cls]
        print(f"  class={cls}: {count_map[cls]} -> {target} (+{n_syn} synthetic)")
    print(f"  total synthetic: {sum(v - count_map[c] for c, v in strategy.items())}")

    return strategy


class SMOTEResampler:
    """
    Thin wrapper around imblearn.SMOTE that fits the same interface used
    throughout this module: resample(X, y) -> (X_res, y_res).

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbours used by SMOTE to construct synthetic
        samples.  Passed straight to imblearn.SMOTE.
    sampling_strategy : str | float | dict
        Passed to imblearn.SMOTE.  Defaults to 'auto' (balance all minority
        classes to match the majority class count).
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: Any = "auto",
        random_state: int = 42,
        total_budget: Optional[int] = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.total_budget = total_budget
        self._smote: Optional[SMOTE] = None

    # ------------------------------------------------------------------
    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample X/y using SMOTE.

        If sampling_strategy='budget', the total number of synthetic samples
        is capped to self.total_budget, distributed proportionally across
        minority classes by their deficit from the majority class. This
        creates a controlled comparison against DenseGAAL.

        If any class has fewer than k_neighbors + 1 samples, k_neighbors is
        automatically reduced to (min_class_count - 1) and a warning is
        printed.

        Returns
        -------
        X_res, y_res : np.ndarray
            Resampled feature matrix and label vector.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            warnings.warn(
                "SMOTEResampler.resample: only one class present — "
                "returning original data unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            return X, y

        # Resolve budget-capped strategy before passing to imblearn
        strategy = self.sampling_strategy
        if strategy == "budget":
            if self.total_budget is None or self.total_budget <= 0:
                warnings.warn(
                    "SMOTEResampler: sampling_strategy='budget' requires "
                    "total_budget > 0. Falling back to 'auto'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                strategy = "auto"
            else:
                strategy = compute_budget_capped_strategy(y, self.total_budget)
                if not strategy:
                    return X, y

        min_count = int(counts.min())
        k_eff = self.k_neighbors

        if min_count <= k_eff:
            k_eff = max(1, min_count - 1)
            warnings.warn(
                f"SMOTEResampler: k_neighbors reduced to {k_eff} because the "
                f"smallest class has only {min_count} sample(s).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._smote = SMOTE(
            k_neighbors=k_eff,
            sampling_strategy=strategy,
            random_state=self.random_state,
        )

        try:
            X_res, y_res = self._smote.fit_resample(X, y)
        except ValueError as exc:
            warnings.warn(
                f"SMOTEResampler.resample failed ({exc}); "
                "returning original data unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            return X, y

        print(
            f"[SMOTE] {len(y)} -> {len(y_res)} samples "
            f"(+{len(y_res) - len(y)} synthetic)."
        )
        _print_synthetic_delta(y, y_res, "SMOTE")
        _print_dist("SMOTE result", y_res)
        return X_res, y_res


class ADASYNResampler:
    """
    Thin wrapper around imblearn.ADASYN.

    ADASYN generates more synthetic samples in regions that are harder to
    learn (near the decision boundary), unlike SMOTE which generates samples
    uniformly along line segments.  This makes it a useful comparison point
    for DenseGAAL's density-driven approach.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours used by ADASYN.
    sampling_strategy : str | float | dict
        Passed to imblearn.ADASYN.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        sampling_strategy: Any = "auto",
        random_state: int = 42,
        total_budget: Optional[int] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.total_budget = total_budget
        self._adasyn: Optional[ADASYN] = None

    # ------------------------------------------------------------------
    def resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample X/y using ADASYN.

        If sampling_strategy='budget', the total number of synthetic samples
        is capped to self.total_budget, distributed proportionally across
        minority classes by their deficit from the majority class. This
        creates a controlled comparison against DenseGAAL.

        Falls back to original data if ADASYN raises (e.g. when the labeled
        set is too small or already balanced).

        Returns
        -------
        X_res, y_res : np.ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            warnings.warn(
                "ADASYNResampler.resample: only one class present — "
                "returning original data unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            return X, y

        # Resolve budget-capped strategy before passing to imblearn
        strategy = self.sampling_strategy
        if strategy == "budget":
            if self.total_budget is None or self.total_budget <= 0:
                warnings.warn(
                    "ADASYNResampler: sampling_strategy='budget' requires "
                    "total_budget > 0. Falling back to 'auto'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                strategy = "auto"
            else:
                strategy = compute_budget_capped_strategy(y, self.total_budget)
                if not strategy:
                    return X, y

        # ADASYN uses n_neighbors + 1 internally for density estimation, so
        # any class with fewer than n_neighbors + 2 samples will crash even
        # after we reduce n_eff. We resolve this by converting the strategy to
        # an explicit dict and removing classes that are too small to resample.
        # This is safer than silently returning unchanged data for the whole set.
        n_eff = self.n_neighbors
        min_safe = n_eff + 2   # ADASYN internal requirement

        count_map = dict(zip(classes, counts))

        # Convert string strategies to explicit dict so we can filter per class
        if isinstance(strategy, str):
            maj_count = int(counts.max())
            if strategy == "auto":
                strategy = {
                    cls: int(maj_count)
                    for cls, cnt in count_map.items()
                    if cnt < maj_count
                }
            elif strategy == "not majority":
                maj_cls = classes[np.argmax(counts)]
                strategy = {
                    cls: int(maj_count)
                    for cls, cnt in count_map.items()
                    if cls != maj_cls
                }
            # other string strategies passed through as-is (imblearn handles them)

        if isinstance(strategy, dict):
            skipped = {
                cls: cnt
                for cls, cnt in count_map.items()
                if cls in strategy and cnt < min_safe
            }
            if skipped:
                for cls, cnt in skipped.items():
                    warnings.warn(
                        f"ADASYNResampler: skipping class={cls} ({cnt} sample(s)) — "
                        f"needs at least {min_safe} samples for ADASYN with "
                        f"n_neighbors={n_eff}. This class will not be oversampled.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                strategy = {
                    cls: target
                    for cls, target in strategy.items()
                    if cls not in skipped
                }
            if not strategy:
                warnings.warn(
                    "ADASYNResampler: all minority classes were skipped due to "
                    "insufficient samples. Returning original data unchanged.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return X, y

            # Recompute n_eff based on the smallest remaining eligible class
            eligible_counts = [count_map[cls] for cls in strategy]
            min_eligible = int(min(eligible_counts))
            if min_eligible < min_safe:
                n_eff = max(1, min_eligible - 2)
                warnings.warn(
                    f"ADASYNResampler: n_neighbors reduced to {n_eff} for "
                    f"smallest eligible class ({min_eligible} samples).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._adasyn = ADASYN(
            n_neighbors=n_eff,
            sampling_strategy=strategy,
            random_state=self.random_state,
        )

        try:
            X_res, y_res = self._adasyn.fit_resample(X, y)
        except RuntimeError as exc:
            # ADASYN raises RuntimeError when no neighbors of a minority sample
            # belong to the majority class (density ratio undefined / div by zero).
            # This happens when minority classes are highly isolated in feature space
            # and is a fundamental incompatibility between ADASYN and this dataset,
            # not a software error. We skip augmentation and return the original data
            # so the AL-only result is still recorded cleanly.
            # For reporting: use the corresponding SMOTE result for this configuration
            # since the labeled set and AL function are identical.
            warnings.warn(
                f"ADASYNResampler: ADASYN is not applicable for this configuration "
                f"({exc}). Minority classes are too isolated for density ratio "
                f"estimation. Returning original data unchanged — use the "
                f"corresponding SMOTE result for this AL function instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            print(
                "[ADASYN] Skipping augmentation — ADASYN not applicable for this "
                "dataset/AL function combination. See warning above."
            )
            return X, y
        except ValueError as exc:
            warnings.warn(
                f"ADASYNResampler.resample failed ({exc}); "
                "returning original data unchanged.",
                RuntimeWarning,
                stacklevel=2,
            )
            return X, y

        print(
            f"[ADASYN] {len(y)} -> {len(y_res)} samples "
            f"(+{len(y_res) - len(y)} synthetic)."
        )
        _print_synthetic_delta(y, y_res, "ADASYN")
        _print_dist("ADASYN result", y_res)
        return X_res, y_res


# ===========================================================================
# 2.  Helpers (kept local to avoid circular imports with the main runner)
# ===========================================================================

def _print_dist(title: str, y: np.ndarray) -> None:
    u, c = np.unique(y, return_counts=True)
    print(f"{title}:")
    for ui, ci in zip(u, c):
        print(f"  Class {ui}: {ci} ({ci / len(y):.2%})")
    print()


def _print_synthetic_delta(y_before: np.ndarray, y_after: np.ndarray, tag: str) -> None:
    """
    Print the number of synthetic samples added per class and in total,
    comparing the label distribution before and after resampling.
    """
    before_counts = dict(zip(*np.unique(y_before, return_counts=True)))
    after_counts  = dict(zip(*np.unique(y_after,  return_counts=True)))
    all_classes   = sorted(set(before_counts) | set(after_counts))

    print(f"[{tag}] synthetic samples added per class:")
    total_added = 0
    for cls in all_classes:
        n_before = int(before_counts.get(cls, 0))
        n_after  = int(after_counts.get(cls, 0))
        delta    = n_after - n_before
        total_added += delta
        print(f"  class={cls}: {n_before} -> {n_after} (+{delta} synthetic)")
    print(f"[{tag}] total synthetic samples added: {total_added}\n")


def _load_data(cfg):
    """Load train/val/test npz splits exactly as in the main runner."""
    def _load(p):
        with np.load(p, allow_pickle=True) as d:
            return d["feature"], d["label"]

    base = cfg.DATASET.DATA_DIR
    return (
        *_load(os.path.join(base, cfg.DATASET.TRAIN_FILE)),
        *_load(os.path.join(base, cfg.DATASET.VAL_FILE)),
        *_load(os.path.join(base, cfg.DATASET.TEST_FILE)),
    )


def _init_classifier(name: str, args, cfg):
    """Mirror of `init_classifier` in the main runner."""
    if name == "MLP":
        return TorchMLPClassifier(
            cfg,
            hidden_layer_sizes=(100,),
            max_iter=100,
            batch_size=64,
            lr=1e-3,
            random_state=args.random_state,
            device="cuda",
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=100, random_state=args.random_state, n_jobs=-1
        )
    if name == "XGBC":
        return XGBClassifier(
            use_label_encoder=False,
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            eval_metric="mlogloss",
            random_state=args.random_state,
        )
    raise ValueError(f"Unknown classifier: {name}")


def _init_pooling(name: str, Xl, yl, Xu, yu):
    """Mirror of `init_pooling` in the main runner."""
    if name == "anchoral":
        return subpool_anchoral(Xl, yl, Xu, yu, M=int(round(Xu.shape[0] * 0.40)))
    if name == "randsub":
        return subpool_randsub(Xl, yl, Xu, yu, M=int(round(Xu.shape[0] * 0.40)))
    if name == "seals":
        return subpool_seals(Xl, yl, Xu, yu)
    raise ValueError(f"Unknown pooling method: {name}")


def _compute_metrics(y_pred, y_true) -> Tuple[Dict, str]:
    """Mirror of `compute_metrics` in the main runner."""
    rpt = classification_report(y_true, y_pred, digits=4)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }, rpt


def _active_learning_loops(args, Xtr, ytr, Xv, yv, clf, Xt=None, yt=None):
    """
    Exact copy of `active_learning_loops` from the main runner so this
    module is self-contained when imported.  If you import from the main
    runner directly, replace this with:

        from main_runner import active_learning_loops as _active_learning_loops
    """
    idx_pool = np.arange(len(yv))
    all_sel = []
    step_pool_f1: list[float] = []
    step_test_f1: list[float] = []

    for step in range(max(0, int(args.al_steps))):
        if len(Xv) == 0:
            print(f"[AL] step {step + 1}: pool empty, stopping.")
            break

        clf.fit(Xtr, ytr)
        al_fn = METHOD_DICT[args.al_function]()
        sel = al_fn.sample(Xv, args.budget, clf, Xtr)

        if sel is None or len(sel) == 0:
            print(f"[AL] step {step + 1}: no selection, stopping.")
            break

        u, c = np.unique(yv[sel], return_counts=True)
        print(f"[AL] step {step + 1}/{args.al_steps} selected:", dict(zip(u, c)))

        Xtr = np.vstack([Xtr, Xv[sel]])
        ytr = np.hstack([ytr, yv[sel]])
        all_sel.append(idx_pool[sel])

        mask = np.ones(len(Xv), dtype=bool)
        mask[sel] = False
        Xv, yv = Xv[mask], yv[mask]
        idx_pool = idx_pool[mask]

        clf.fit(Xtr, ytr)
        if len(Xv):
            f1 = f1_score(yv, clf.predict(Xv), average="macro", zero_division=0)
            print(f"[AL] step {step + 1}: pool macro F1={f1:.4f}")
            step_pool_f1.append(float(f1))
        if Xt is not None and yt is not None:
            f1 = f1_score(yt, clf.predict(Xt), average="macro", zero_division=0)
            print(f"[AL] step {step + 1}: test macro F1={f1:.4f}")
            step_test_f1.append(float(f1))

    sel_all = np.hstack(all_sel).astype(int) if all_sel else np.array([], dtype=int)
    return Xtr, ytr, Xv, yv, sel_all, step_pool_f1, step_test_f1


def _ensure_results_dir(args, resampler_tag: str) -> str:
    """
    Build a results directory path consistent with the main runner's
    `ensure_results_dir`, with resampler and sampling strategy tags so
    runs never collide with each other or with generative-augmentation runs.
    """
    pool = args.pooling_method if args.pooling_method else ""
    clf = args.classifier
    budget_tag = f"budget{args.budget}"
    seed_tag = f"seed{args.random_state}"
    # Include sampling strategy in path so auto vs budget runs are separated
    strategy = getattr(args, "sampling_strategy", "auto")
    strategy_tag = f"strategy_{strategy}" if isinstance(strategy, str) else "strategy_custom"
    path = os.path.join(
        "results",
        args.dataset,
        pool,
        resampler_tag,          # e.g. "SMOTE", "ADASYN", "SMOTE+AnchorAL"
        strategy_tag,           # e.g. "strategy_auto", "strategy_budget"
        clf,
        budget_tag,
        seed_tag,
        args.al_function,
    )
    os.makedirs(path, exist_ok=True)
    return path


def _base_setup(args):
    """Load data, standardise, and fit initial classifier — mirrors `base_set_up`."""
    cfg = get_config(args.dataset, args.al_method)
    Xtr, ytr, Xv, yv, Xt, yt = _load_data(cfg)
    _print_dist("Train (initial labeled)", ytr)
    _print_dist("Val (unlabeled pool)", yv)
    _print_dist("Test", yt)

    if cfg.DATASET.STANDARDIZE:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xv = scaler.transform(Xv)
        Xt = scaler.transform(Xt)

    clf = _init_classifier(args.classifier, args, cfg)
    clf.fit(Xtr, ytr)
    return cfg, clf, Xtr, ytr, Xv, yv, Xt, yt


def _write_report(res_dir: str, rpt: str, step_metrics: Dict, args, elapsed: float, resampler_tag: str) -> None:
    """Write classification report + metadata, mirroring the main runner."""
    with open(os.path.join(res_dir, "report.txt"), "w") as f:
        f.write(rpt)
        al_pool_f1 = step_metrics.get("al_pool_f1", [])
        al_test_f1 = step_metrics.get("al_test_f1", [])
        resamp_test_f1 = step_metrics.get("resamp_test_f1", [])
        if al_pool_f1 or al_test_f1 or resamp_test_f1:
            f.write("\n\n[Step metrics]\n")
        if al_pool_f1:
            f.write("al_pool_macro_f1:\n")
            for i, v in enumerate(al_pool_f1, 1):
                f.write(f"  step_{i}: {v:.4f}\n")
        if al_test_f1:
            f.write("al_test_macro_f1:\n")
            for i, v in enumerate(al_test_f1, 1):
                f.write(f"  step_{i}: {v:.4f}\n")
        if resamp_test_f1:
            f.write(f"{resampler_tag}_test_macro_f1:\n")
            for i, v in enumerate(resamp_test_f1, 1):
                f.write(f"  step_{i}: {v:.4f}\n")
        f.write(
            "\n\n[Run metadata]\n"
            f"resampler: {resampler_tag}\n"
            f"elapsed_seconds: {elapsed:.4f}\n"
            f"random_state: {args.random_state}\n"
            f"sampling_strategy: {args.sampling_strategy}\n"
        )


# ===========================================================================
# 3.  Core runner — shared by SMOTE and ADASYN
# ===========================================================================

def _run_resampler_baseline(
    args,
    resampler,               # SMOTEResampler | ADASYNResampler instance
    resampler_tag: str,      # used for logging and results-dir naming
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generic runner that:
      1. Runs active learning loops (respects --al_steps, --pooling_method).
      2. Applies the supplied resampler to the resulting labeled set.
      3. Trains the final classifier on the augmented set.
      4. Evaluates on the test set.

    Returns
    -------
    y_pred, y_true, step_metrics
        Same contract as run_base / run_augmented / run_alfa in the main runner.
    """
    res_dir = _ensure_results_dir(args, resampler_tag)
    print(f"\n{'='*60}")
    print(f"Resampler baseline: {resampler_tag}")
    print(f"Results dir: {res_dir}")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = _base_setup(args)

    # Optionally apply a set-construction pooling strategy before AL
    if args.pooling_method:
        print(f"[{resampler_tag}] Applying pooling: {args.pooling_method}")
        Xv, yv = _init_pooling(args.pooling_method, Xtr, ytr, Xv, yv)

    # --- Active learning phase ------------------------------------------
    Xal, yal, _Xv, _yv, _sel, step_pool_f1, step_test_f1 = _active_learning_loops(
        args, Xtr, ytr, Xv, yv, clf, Xt, yt
    )

    print(f"\n[{resampler_tag}] AL complete.")
    _print_dist("Labeled set after AL", yal)

    # --- Resampling phase -----------------------------------------------
    print(f"\n[{resampler_tag}] Applying {resampler_tag} to labeled set ...")
    Xf, yf = resampler.resample(Xal, yal)

    # --- Final classifier -----------------------------------------------
    print(f"\n[{resampler_tag}] Training final classifier on augmented set ...")
    clf.fit(Xf, yf)
    y_pred = clf.predict(Xt)

    f1_macro = f1_score(yt, y_pred, average="macro", zero_division=0)
    print(f"[{resampler_tag}] Test macro F1: {f1_macro:.4f}")

    elapsed = time.perf_counter() - start_time
    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    step_metrics: Dict = {
        "al_pool_f1": step_pool_f1,
        "al_test_f1": step_test_f1,
        "resamp_test_f1": [float(f1_macro)],
    }

    metrics, rpt = _compute_metrics(y_pred, yt)
    _write_report(res_dir, rpt, step_metrics, args, elapsed, resampler_tag)

    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                **metrics,
                "elapsed_seconds": elapsed,
                "peak_gpu_mem_mb": peak_gpu_mb,
                "resampler": resampler_tag,
            },
            f,
            indent=4,
        )

    print(f"\n[{resampler_tag}] Report saved to: {res_dir}")
    return y_pred, yt, step_metrics


# ===========================================================================
# 4.  Public API — one function per baseline (matches main runner conventions)
# ===========================================================================

def run_smote_baseline(args) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    SMOTE augmentation on top of active learning.

    Mirrors the signature of `run_augmented` / `run_base` in the main runner
    so it can be dropped into the dispatch in `main()` with no further changes.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain at minimum:
        al_method, al_function, classifier, dataset, budget, random_state,
        al_steps, pooling_method, smote_k_neighbors, sampling_strategy.

        When sampling_strategy='budget', total synthetic samples are capped to
        budget * al_steps, distributed proportionally across minority classes.

    Returns
    -------
    y_pred, y_true, step_metrics
    """
    total_budget = int(args.budget) * int(getattr(args, "al_steps", 1))
    resampler = SMOTEResampler(
        k_neighbors=getattr(args, "smote_k_neighbors", 5),
        sampling_strategy=getattr(args, "sampling_strategy", "auto"),
        random_state=args.random_state,
        total_budget=total_budget,
    )
    tag = "SMOTE+AnchorAL" if getattr(args, "pooling_method", None) == "anchoral" else "SMOTE"
    return _run_resampler_baseline(args, resampler, tag)


def run_adasyn_baseline(args) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    ADASYN augmentation on top of active learning.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain at minimum:
        al_method, al_function, classifier, dataset, budget, random_state,
        al_steps, pooling_method, adasyn_n_neighbors, sampling_strategy.

        When sampling_strategy='budget', total synthetic samples are capped to
        budget * al_steps, distributed proportionally across minority classes.

    Returns
    -------
    y_pred, y_true, step_metrics
    """
    total_budget = int(args.budget) * int(getattr(args, "al_steps", 1))
    resampler = ADASYNResampler(
        n_neighbors=getattr(args, "adasyn_n_neighbors", 5),
        sampling_strategy=getattr(args, "sampling_strategy", "auto"),
        random_state=args.random_state,
        total_budget=total_budget,
    )
    tag = "ADASYN+AnchorAL" if getattr(args, "pooling_method", None) == "anchoral" else "ADASYN"
    return _run_resampler_baseline(args, resampler, tag)


# ===========================================================================
# 5.  CLI — standalone execution for quick testing
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMOTE / ADASYN baselines for DenseGAAL active learning pipeline."
    )

    # ------------------------------------------------------------------
    # Flags shared with the main runner (subset needed here)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--al_method",
        choices=["base", "DA", "DA+ALFA"],
        default="base",
        help="Active learning method (only affects pooling logic; augmentation is "
             "always SMOTE or ADASYN when running this script).",
    )
    parser.add_argument(
        "--pooling_method",
        choices=["anchoral", "randsub", "seals"],
        default=None,
        help="Optional set-construction strategy applied before AL acquisition. "
             "Use 'anchoral' to reproduce the SMOTE+AnchorAL condition.",
    )
    parser.add_argument(
        "--al_function",
        choices=list(METHOD_DICT.keys()),
        required=True,
        help="Point acquisition function (e.g. margin, entropy, clue).",
    )
    parser.add_argument(
        "--classifier",
        choices=["MLP", "RF", "XGBC"],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--al_steps",
        type=int,
        default=1,
        help="Number of AL acquisition rounds before resampling.",
    )

    # ------------------------------------------------------------------
    # Resampler flags (new)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--resampler",
        choices=["SMOTE", "ADASYN"],
        required=True,
        help="Which resampler to apply after active learning.",
    )
    parser.add_argument(
        "--smote_k_neighbors",
        type=int,
        default=5,
        help="k for SMOTE nearest-neighbour interpolation (default: 5).",
    )
    parser.add_argument(
        "--adasyn_n_neighbors",
        type=int,
        default=5,
        help="k for ADASYN density estimation (default: 5).",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="auto",
        help=(
            "Sampling strategy passed to imbalanced-learn. "
            "'auto' balances all minority classes to match the majority count. "
            "'not majority' resamples all but the largest class. "
            "'budget' caps total synthetic samples to budget * al_steps, "
            "distributed proportionally across minority classes by deficit — "
            "use this for a controlled comparison against DenseGAAL. "
            "A float sets the minority-to-majority ratio after resampling. "
            "A JSON string encodes a {class: target_count} dict."
        ),
    )

    args = parser.parse_args()

    # Parse sampling_strategy: allow JSON dict or float from CLI,
    # but keep string keywords ('auto', 'not majority', 'budget', etc.) as-is.
    strat = args.sampling_strategy
    if strat not in ("auto", "not majority", "not minority", "all", "minority", "budget"):
        try:
            strat = float(strat)
        except ValueError:
            try:
                import json as _json
                strat = _json.loads(strat)
            except Exception:
                pass  # keep as string; imbalanced-learn will validate
    args.sampling_strategy = strat

    return args


def main() -> None:
    args = _parse_args()

    if args.resampler == "SMOTE":
        y_pred, y_true, step_metrics = run_smote_baseline(args)
    elif args.resampler == "ADASYN":
        y_pred, y_true, step_metrics = run_adasyn_baseline(args)
    else:
        raise ValueError(f"Unknown resampler: {args.resampler}")

    metrics, rpt = _compute_metrics(y_pred, y_true)
    print("\n" + rpt)
    print("Macro F1:", metrics["f1_macro"])


if __name__ == "__main__":
    main()