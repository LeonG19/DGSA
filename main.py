#!/usr/bin/env python3
"""
Unified Active Learning runner with pluggable generators:
- Active Learning methods (`--al_method`): base, DA, DenseGAAL
- Active Learning functions (`--al_function`): specific query functions (e.g. margin, powermargin)
- Generators (`--generator`): TVAE, CTGAN, RTF
- Classifiers (`--classifier`): MLP, RF, XGBC

This script creates a results directory (`results/{dataset}/.../budget{budget}/...`)
and saves classification reports and detailed metrics (accuracy, precision, recall, F1 — micro & macro).
Supports custom anchor fraction via `--anchor_alpha` and `--anchor_steepness`.

FLAGS:
- --neighbor_only: in DenseGAAL, add kNN neighbors directly (no generator / no synthetic data)
- --filter_bad_neighbors: remove incorrectly-labeled neighbors (uses pool ground-truth labels y_p)
- --no_local_support: in DenseGAAL, SKIP kNN neighbors; train generator using anchors only
- --gen_train_all_labeled: in DenseGAAL, train generator on (all labeled so far) + (retrieved neighbors)
  instead of only (anchors + neighbors). Works in minority and base DenseGAAL.

FLAGS for iteration control:
 - --al_steps: number of active learning loops to run before augmentation (default: 1, uses full budget per step).
 - --densegaal_steps: number of DenseGAAL iterations to run (default: 1). Each iteration runs 1 AL step + DenseGAAL.
 - --densegaal_steps_mode: when --densegaal_steps>1, choose whether to run DenseGAAL after EACH AL step
   (per_step, default) or to run ALL AL steps first and generate ONCE (per_al_steps).

UPDATED FLAG BEHAVIOR:
- --num_synthetic now supports a LIST of multipliers:
    * If 1 number: used for all augmentable classes
    * If k>1 numbers: applied to the k most underrepresented classes (ascending by labeled count),
      and the rest get multiplier 0 (no synthetic) by default.
  This also works in --minority mode (which uses the 2 most underrepresented classes).
"""

import argparse
import os
import math
import json
import time
from collections import Counter

from pool_filters_cumulative import subpool_anchoral, subpool_randsub, subpool_seals
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus

from active_learning_functions import METHOD_DICT
from config import get_config
from ctgan import TVAE
from ctgan import CTGAN
from realtabformer import REaLTabFormer
from classifiers.mlp import TorchMLPClassifier
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Unified AL runner")
    parser.add_argument('--al_method', choices=['base','DA','DenseGAAL'], required=True)
    parser.add_argument('--pooling_method', choices=['anchoral', 'randsub', 'seals'], required=False, default=False)
    parser.add_argument('--al_function', choices=list(METHOD_DICT.keys()), required=True)
    parser.add_argument('--generator', choices=['TVAE','CTGAN','RTF'])
    parser.add_argument('--classifier', choices=['MLP','RF','XGBC'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--al_steps', type=int, default=1,
                        help="Number of active learning loops to run before augmentation (default: 1). "
                             "Each step uses the full --budget.")
    parser.add_argument('--densegaal_steps', type=int, default=1,
                        help="Number of DenseGAAL iterations to run (default: 1). "
                             "Each iteration runs 1 AL step + DenseGAAL.")
    parser.add_argument('--densegaal_steps_mode', choices=['per_step', 'per_al_steps'], default='per_step',
                        help="When --densegaal_steps>1: run DenseGAAL after EACH AL step (per_step, default) "
                             "or run ALL AL steps first and generate ONCE (per_al_steps).")
    parser.add_argument('--num_synthetic', nargs='+', type=str, default=['1'])
    parser.add_argument('--decay_power', type=float, default=0.0)
    parser.add_argument('--filter_synthetic', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--steepness', type=float, default=100.0)
    parser.add_argument('--minority', action='store_true')
    parser.add_argument('--neighbor_only', action='store_true')
    parser.add_argument('--filter_bad_neighbors', action='store_true')
    parser.add_argument('--no_local_support', action='store_true',
                        help="DenseGAAL mode: skip kNN neighbors; train generator using anchors only.")
    parser.add_argument('--gen_train_all_labeled', action='store_true',
                        help="Train generator on (all labeled so far) + (retrieved neighbors) "
                             "instead of (anchors + neighbors).")


    parser.add_argument('--rep', choices=['raw','scaled','pca'], default='scaled',
                    help="Representation/encoding used for anchor retrieval + ranking. "
                         "raw=use current feature space; scaled=StandardScaler; "
                         "pca=StandardScaler+PCA.")

    parser.add_argument('--metric', choices=['cosine','euclidean'], default='euclidean',
                    help="Distance metric in representation space (cosine recommended).")

    parser.add_argument('--pca_dim', type=int, default=64,
                    help="PCA dimension when --rep pca.")

    parser.add_argument('--anchoral_A', type=int, default=10,
                    help="Number of anchors per class (A) for densegaal_anchoral mode (kmeans++ seeds).")
    parser.add_argument('--anchoral_K', type=int, default=1,
                    help="K neighbors to retrieve per anchor (K) for densegaal_anchoral mode.")
    parser.add_argument('--anchoral_M', type=int, default=2600,
                    help="Top-M unlabeled points to keep after ranking (AnchorAL subpool size).")
    parser.add_argument('--alfa_k', type=int, default=1,
                    help="K neighbors to retrieve per anchor for ORIGINAL DenseGAAL (non-anchoral modes).")

    parser.add_argument('--anchoral_anchor_total', type=int, default=None,
                help="Total number of anchors when --anchoral_anchor_policy inverse_rank. "
                     "If omitted, defaults to 50. Ignored when policy=per_class.")



    return args


def _parse_num_synthetic_list(raw_tokens):
    """
    Accepts:
      raw_tokens like ['3'] or ['5','3'] or ['6,4,2'] or ['6,4','2']
    Returns: list[float]
    """
    if raw_tokens is None:
        return [3.0]
    out = []
    for tok in raw_tokens:
        if tok is None:
            continue
        parts = [p.strip() for p in str(tok).split(',') if p.strip() != '']
        for p in parts:
            out.append(float(p))
    if len(out) == 0:
        out = [3.0]
    return out


def _rank_classes_by_underrep(y, classes):
    """
    Return list of classes sorted by ascending count in y (most underrepresented first).
    Only ranks within the provided `classes`.
    """
    classes = list(classes)
    counts = {cls: int(np.sum(y == cls)) for cls in classes}
    return sorted(classes, key=lambda cls: (counts[cls], str(cls)))


def _make_multiplier_map(y_labeled, classes_to_aug, multipliers):
    """
    multipliers:
      - len==1: apply to all classes_to_aug
      - len>1: apply to k most underrepresented classes (k=len(multipliers)), others -> 0.0
    """
    multipliers = list(multipliers)
    classes_to_aug = list(classes_to_aug)

    if len(classes_to_aug) == 0:
        return {}

    if len(multipliers) == 1:
        return {cls: float(multipliers[0]) for cls in classes_to_aug}

    ranked = _rank_classes_by_underrep(y_labeled, classes_to_aug)
    m = {cls: 0.0 for cls in classes_to_aug}
    for i, mult in enumerate(multipliers):
        if i >= len(ranked):
            break
        m[ranked[i]] = float(mult)
    return m


def ensure_results_dir(args):
    gen = args.generator if args.generator else ''
    clf = args.classifier if args.classifier else ''
    pool = args.pooling_method if args.pooling_method else ''
    minority = "minority" if args.minority else ''
    knn_only = "knnonly" if args.neighbor_only else ''
    knn_filter = "filter" if args.filter_bad_neighbors else ''
    all_anchors = ''
    no_local = "nolocalsupport" if getattr(args, 'no_local_support', False) else ''
    rep_tag = f"rep-{args.rep}_{args.metric}" if hasattr(args, 'rep') else ''
    anch_hybrid = "anchoral_densegaal" if getattr(args, 'densegaal_anchoral', False) else ''
    if getattr(args, 'no_local_support', False):
        neighbors_k = 0
    else:
        neighbors_k = args.anchoral_K if getattr(args, 'densegaal_anchoral', False) else args.alfa_k
    neighbors_tag = f"neighborsK{neighbors_k}"
    al_steps_tag = f"alsteps{args.al_steps}" if getattr(args, 'al_steps', 1) != 0 else ''
    densegaal_steps_tag = f"densegaalsteps{args.densegaal_steps}" if getattr(args, 'densegaal_steps', 1) != 0 else ''
    densegaal_steps_mode_tag = f"densegaalmode{args.densegaal_steps_mode}" if getattr(args, 'densegaal_steps_mode', 'per_step') != 'per_step' else ''
    worst_tag = ''
    seed_tag = f"seed{args.random_state}"
    budget_tag = f"budget{args.budget}"
    decay_power_tag = f"decaypower{args.decay_power}"
    steepness_tag = f"steepness{args.steepness}"

    # encode these modes in path so you can separate runs cleanly
    all_labeled = "alllabeled" if args.gen_train_all_labeled else ''

    path = os.path.join(
        'results', args.dataset, pool, args.al_method, clf, gen,
        budget_tag, decay_power_tag, steepness_tag,
        minority, all_labeled, all_anchors, no_local, anch_hybrid, rep_tag, neighbors_tag, al_steps_tag, densegaal_steps_tag, densegaal_steps_mode_tag, worst_tag, seed_tag,
        f"{args.al_function}_{knn_only}_{knn_filter}"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _dataset_versions_dir(results_dir):
    return os.path.join(results_dir, "dataset_versions")


def _label_name(cfg):
    return getattr(cfg.DATASET, "LABEL_NAME", "Label")


def _feature_names_for_export(cfg, X):
    fn = list(getattr(cfg.DATASET, "FEATURE_NAMES", []))
    if not fn or len(fn) != X.shape[1]:
        fn = [f"feature_{i}" for i in range(X.shape[1])]
    return fn


def _write_csv(path, X, y, feature_names, label_name):
    df = pd.DataFrame(X, columns=feature_names)
    df[label_name] = y
    df.to_csv(path, index=False)
    return True


def export_dataset_versions(
    args,
    cfg,
    results_dir,
    Xtr,
    ytr,
    Xv,
    yv,
    Xt,
    yt,
    Xal=None,
    yal=None,
    Xaug=None,
    yaug=None,
):
    out_dir = _dataset_versions_dir(results_dir)
    os.makedirs(out_dir, exist_ok=True)
    feature_names = _feature_names_for_export(cfg, Xtr)
    label_name = _label_name(cfg)

    _write_csv(os.path.join(out_dir, "train.csv"), Xtr, ytr, feature_names, label_name)
    _write_csv(os.path.join(out_dir, "val.csv"), Xv, yv, feature_names, label_name)
    _write_csv(os.path.join(out_dir, "test.csv"), Xt, yt, feature_names, label_name)

    if Xal is not None and yal is not None:
        al_changed = Xal.shape[0] != Xtr.shape[0]
        if int(getattr(args, "al_steps", 0)) > 0 or al_changed:
            _write_csv(os.path.join(out_dir, "train+al.csv"), Xal, yal, feature_names, label_name)

    if Xaug is not None and yaug is not None:
        _write_csv(os.path.join(out_dir, "train+gaal.csv"), Xaug, yaug, feature_names, label_name)


def _init_al_method(args):
    """
    Initialize AL method with its own defaults.
    """
    cls = METHOD_DICT[args.al_function]
    return cls()


def load_data(cfg):
    def _load(p):
        with np.load(p, allow_pickle=True) as d:
            return d['feature'], d['label']
    base = cfg.DATASET.DATA_DIR
    return (*_load(os.path.join(base, cfg.DATASET.TRAIN_FILE)),
            *_load(os.path.join(base, cfg.DATASET.VAL_FILE)),
            *_load(os.path.join(base, cfg.DATASET.TEST_FILE)))


def print_dist(title, y):
    u, c = np.unique(y, return_counts=True)
    print(f"{title}:")
    for ui, ci in zip(u, c):
        print(f"  Class {ui}: {ci} ({ci/len(y):.2%})")
    print()


def compute_anchor_fraction(f_c, min_frac=0.01, alpha=1, steepness=100):
    print(f"Computing anchor fraction for freq={f_c} with min_frac={min_frac}, alpha={alpha}, steepness={steepness}")
    frac = (alpha * (math.exp((-steepness) * (f_c - min_frac))))
    print(f"freq={f_c:.4f} -> frac={frac:.4f}")
    return min(1, frac)




def compute_synthetic_count(
    n_labeled_c: int,
    freq_c: float,          
    freq_min: float,        
    anchor_frac: float = 1.0,
    decay_power: float = 4.0,
    min_round: float = 0.5,
):

    if n_labeled_c <= 0 or freq_min <= 0 or freq_c <= 0:
        return 0

    ratio = max(1.0, freq_c / freq_min)
    weight = ratio ** (-decay_power)  

    expected = n_labeled_c * anchor_frac * weight

    if expected < min_round:
        return 0

    return int(round(expected))


def compute_k_per_cls(
    y_labeled: np.ndarray,
    classes_to_aug,
    *,
    alpha: float,
    steepness: float,
    decay_power: float,
    multipliers,
    min_frac: float = 0.01,
):
    """
    Compute per-class synthetic counts using the same rule everywhere:

      1) anchor_frac = compute_anchor_fraction(freq_c, min_frac, alpha, steepness)
      2) base_k = compute_synthetic_count(n_labeled_c, freq_c, freq_min, anchor_frac, decay_power)
      3) apply optional multiplier map (backward compatible with --num_synthetic behavior)

    Notes:
      - If multipliers has length 1: applied to all classes_to_aug.
      - If multipliers has length >1: applied to the k most underrepresented classes; others -> 0.
    """
    y_labeled = np.asarray(y_labeled)
    classes_to_aug = list(classes_to_aug)
    if len(classes_to_aug) == 0:
        return {}

    # frequency computed from CURRENT labeled set restricted to classes_to_aug
    counts = {cls: int(np.sum(y_labeled == cls)) for cls in classes_to_aug}
    total = int(sum(counts.values()))
    if total <= 0:
        return {cls: 0 for cls in classes_to_aug}

    freqs = {cls: (counts[cls] / total) for cls in classes_to_aug}
    freq_min = max(1e-12, min(freqs.values()))

    mult_map = _make_multiplier_map(y_labeled, classes_to_aug, multipliers) if multipliers is not None else {cls: 1.0 for cls in classes_to_aug}

    k_per_cls = {}
    for cls in classes_to_aug:
        n_c = counts[cls]
        f_c = freqs[cls]
        frac = compute_anchor_fraction(f_c, min_frac=min_frac, alpha=alpha, steepness=steepness)

        base_k = compute_synthetic_count(
            n_labeled_c=n_c,
            freq_c=f_c,
            freq_min=freq_min,
            anchor_frac=frac,
            decay_power=decay_power,
        )

        mult = float(mult_map.get(cls, 0.0)) if mult_map is not None else 1.0
        k = int(round(base_k * mult))
        if k < 0:
            k = 0
        k_per_cls[cls] = k

    return k_per_cls


def init_pooling(name, Xl, yl, Xu, yu):
    if name == 'anchoral':
        print("subpol size", Xu.shape[0]//10)
        return subpool_anchoral(Xl, yl, Xu, yu, M = int(round(Xu.shape[0] * 0.40)))
    elif name == 'randsub':
        print("subpol size", int(round(Xu.shape[0] * 0.40)))

        return subpool_randsub(Xl, yl, Xu, yu, M = int(round(Xu.shape[0] * 0.40)))
    elif name == "seals":
        return subpool_seals(Xl, yl, Xu, yu)




def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z / n


class _RepEncoder:
    """Fit/transform interface for representation used in anchor retrieval."""
    def fit(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray, cfg):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)


class _RawEncoder(_RepEncoder):
    pass


class _ScaledEncoder(_RepEncoder):
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.scaler = None

    def fit(self, X_l, y_l, X_u, cfg):
        # Fit on combined to avoid train/pool scale mismatch
        X_all = np.vstack([X_l, X_u])
        self.scaler = StandardScaler().fit(X_all)
        return self

    def transform(self, X):
        Z = self.scaler.transform(np.asarray(X))
        if self.metric == "cosine":
            Z = _normalize_rows(Z)
        return Z


class _PCAEncoder(_RepEncoder):
    def __init__(self, pca_dim: int = 64, metric: str = "cosine", random_state: int = 42):
        self.pca_dim = pca_dim
        self.metric = metric
        self.random_state = random_state
        self.scaler = None
        self.pca = None

    def fit(self, X_l, y_l, X_u, cfg):
        X_all = np.vstack([X_l, X_u])
        self.scaler = StandardScaler().fit(X_all)
        Xs = self.scaler.transform(X_all)
        self.pca = PCA(n_components=min(self.pca_dim, Xs.shape[1]), random_state=self.random_state).fit(Xs)
        return self

    def transform(self, X):
        Xs = self.scaler.transform(np.asarray(X))
        Z = self.pca.transform(Xs)
        if self.metric == "cosine":
            Z = _normalize_rows(Z)
        return Z




def _make_encoder(args, cfg) -> _RepEncoder:
    if args.rep == "raw":
        return _RawEncoder()
    if args.rep == "scaled":
        return _ScaledEncoder(metric=args.metric)
    if args.rep == "pca":
        return _PCAEncoder(pca_dim=args.pca_dim, metric=args.metric, random_state=args.random_state)
    raise ValueError(f"Unknown rep={args.rep}")


def _compute_anchor_counts(
    y_l: np.ndarray,
    *,
    policy: str,
    total_anchors: int,
    A_per_class: int,
    anchor_alpha: float = 1.0,
    anchor_steepness: float = 10.0,
    random_state: int,
):
    """
    Decide how many anchors to take per class from the CURRENT labeled distribution.

    policies:
      - per_class: take A_per_class anchors per class (previous behavior).
      - densegaal_freq: allocate anchors using the DenseGAAL anchor-fraction function.
          For each class c with labeled count n_c and labeled frequency f_c:
            frac_c = compute_anchor_fraction(f_c, anchor_alpha, anchor_steepness)
          If total_anchors is None/<=0:  A_c = clip(round(frac_c * n_c), 1, n_c)
          If total_anchors > 0: distribute total_anchors across classes proportionally to
            weight_c = max(frac_c, 0)  (with a tiny epsilon to avoid all-zero)
          (Always capped by available labeled samples per class.)
      - inverse_rank: legacy option (kept for backwards compatibility).

    Returns:
      dict {class_label: n_anchors}
    """
    y_l = np.asarray(y_l)
    classes, counts = np.unique(y_l, return_counts=True)

    if len(classes) == 0:
        return {}

    if policy == "per_class":
        return {cls: int(A_per_class) for cls in classes}

    if policy == "alfa_freq":
        # Use DenseGAAL anchor fraction to decide anchor budget per class.
        N = int(counts.sum())
        freqs = {cls: (cnt / N) for cls, cnt in zip(classes, counts)}

        fracs = {cls: float(compute_anchor_fraction(freqs[cls], anchor_alpha, anchor_steepness)) for cls in classes}

        # Case 1: no global anchor budget -> per-class anchors ~= frac * n_c
        if total_anchors is None or int(total_anchors) <= 0:
            out = {}
            for cls, cnt in zip(classes, counts):
                a = int(np.round(fracs[cls] * float(cnt)))
                a = max(1, min(int(cnt), a))
                out[cls] = a
            return out

        # Case 2: distribute a global total_anchors using ALFA fractions as weights.
        total_anchors = int(total_anchors)
        if total_anchors <= 0:
            return {cls: 0 for cls in classes}

        # Weights from fracs; add epsilon to avoid all-zero.
        w = np.array([max(fracs[cls], 0.0) for cls in classes], dtype=float)
        if np.all(w <= 0):
            w = np.ones_like(w)
        w = w / w.sum()

        raw = w * total_anchors
        base = np.floor(raw).astype(int)
        rem = int(total_anchors - base.sum())

        frac = raw - base
        frac_order = np.argsort(-frac)
        for i in range(rem):
            base[frac_order[i % len(base)]] += 1

        # Enforce at least 1 per class when possible, then cap by available labeled counts.
        out = {cls: int(base[i]) for i, cls in enumerate(classes)}

        if total_anchors >= len(classes):
            for cls in classes:
                if out.get(cls, 0) == 0:
                    donor = max(out.keys(), key=lambda c: out[c])
                    if out[donor] > 1:
                        out[donor] -= 1
                        out[cls] = 1

        # Cap by availability and re-balance any overflow/deficit.
        caps = {cls: int(cnt) for cls, cnt in zip(classes, counts)}

        # First cap.
        overflow = 0
        for cls in classes:
            if out[cls] > caps[cls]:
                overflow += (out[cls] - caps[cls])
                out[cls] = caps[cls]

        # Redistribute overflow to classes with remaining capacity.
        if overflow > 0:
            # Greedy: give to classes with most remaining cap.
            while overflow > 0:
                receivers = [c for c in classes if out[c] < caps[c]]
                if not receivers:
                    break
                receiver = max(receivers, key=lambda c: (caps[c] - out[c]))
                out[receiver] += 1
                overflow -= 1

        # Final sum fix to match total_anchors when possible.
        s = sum(out.values())
        if s != total_anchors:
            delta = total_anchors - s
            if delta > 0:
                # add where capacity exists
                while delta > 0:
                    receivers = [c for c in classes if out[c] < caps[c]]
                    if not receivers:
                        break
                    receiver = max(receivers, key=lambda c: (caps[c] - out[c]))
                    out[receiver] += 1
                    delta -= 1
            else:
                # remove from largest allocations (but keep at least 1 when possible)
                while delta < 0:
                    donors = [c for c in classes if out[c] > 1]
                    if not donors:
                        break
                    donor = max(donors, key=lambda c: out[c])
                    out[donor] -= 1
                    delta += 1

        return out

    if policy != "inverse_rank":
        raise ValueError(f"Unknown anchor allocation policy: {policy}")

    total_anchors = int(total_anchors)
    if total_anchors <= 0:
        return {cls: 0 for cls in classes}

    # proportions in descending order
    p = counts / counts.sum()
    p_desc = np.sort(p)[::-1]

    # classes sorted by ascending count (minority first)
    cls_inc = classes[np.argsort(counts)]

    # raw targets
    raw = p_desc * total_anchors
    base = np.floor(raw).astype(int)
    rem = int(total_anchors - base.sum())

    # distribute remainder by largest fractional parts
    frac = raw - base
    frac_order = np.argsort(-frac)
    for i in range(rem):
        base[frac_order[i % len(base)]] += 1

    out = {cls_inc[i]: int(base[i]) for i in range(len(cls_inc))}

    # ensure at least 1 per class when possible
    if total_anchors >= len(classes):
        for cls in classes:
            if out.get(cls, 0) == 0:
                donor = max(out.keys(), key=lambda c: out[c])
                if out[donor] > 1:
                    out[donor] -= 1
                    out[cls] = 1

    # final sum fix
    s = sum(out.values())
    if s != total_anchors and len(out):
        donor = max(out.keys(), key=lambda c: out[c])
        out[donor] += (total_anchors - s)

    return out

def _anchoral_subpool_with_pseudolabels(
    Z_l: np.ndarray,
    y_l: np.ndarray,
    Z_u: np.ndarray,
    *,
    A_per_class: int,
    anchor_policy: str,
    anchor_total: int,
    K: int,
    M: int,
    metric: str,
    anchor_alpha: float,
    anchor_steepness: float,
    random_state: int,
):
    """
    AnchorAL-style:
      1) Choose A anchors/class from labeled set using k-means++ in Z_l
      2) For each anchor, retrieve KNN from Z_u
      3) Aggregate similarity per unlabeled point and per anchor-class
      4) Rank by best class similarity; keep top-M
      5) Pseudo-label each selected point by argmax class similarity

    Returns:
      anchor_pos (indices into labeled set)
      idx_u (indices into Z_u)
      y_pseudo (pseudo labels aligned to idx_u)
    """
    rng = np.random.RandomState(random_state)
    y_l = np.asarray(y_l)
    # --- select anchors (positions in labeled array)
    anchor_pos = _anchoral_select_anchors(
        Z_l=Z_l,
        y_l=y_l,
        A_per_class=A_per_class,
        anchor_policy=anchor_policy,
        anchor_total=anchor_total,
        anchor_alpha=anchor_alpha,
        anchor_steepness=anchor_steepness,
        random_state=random_state,
    )
    nn = NearestNeighbors(metric=("cosine" if metric == "cosine" else "euclidean")).fit(Z_u)

    # per unlabeled idx: dict[class -> list[sims]]
    per_u = {}

    for a in anchor_pos:
        cls = y_l[a]
        dists, idxs = nn.kneighbors(Z_l[a][None, :], n_neighbors=min(K, len(Z_u)), return_distance=True)
        dists, idxs = dists[0], idxs[0]
        sims = 1.0 - dists if metric == "cosine" else -dists  # higher is better
        for i, s in zip(idxs.tolist(), sims.tolist()):
            per_u.setdefault(int(i), {}).setdefault(cls, []).append(float(s))

    if not per_u:
        # fallback: random subpool
        idx_u = rng.choice(np.arange(len(Z_u)), size=min(M, len(Z_u)), replace=False)
        y_pseudo = np.full(len(idx_u), np.unique(y_l)[0], dtype=y_l.dtype)
        return anchor_pos, idx_u.astype(int), y_pseudo

    # score each unlabeled point by best class avg sim
    rows = []
    for i, cls_map in per_u.items():
        # avg sim per class
        cls_avg = {c: float(np.mean(v)) for c, v in cls_map.items()}
        # pick best class
        best_c = max(cls_avg.keys(), key=lambda c: cls_avg[c])
        best_s = cls_avg[best_c]
        rows.append((i, best_s, best_c))

    rows = np.array(rows, dtype=object)
    order = np.argsort(-rows[:, 1].astype(float))
    rows = rows[order]
    if M is not None and M > 0:
        rows = rows[: min(M, len(rows))]

    idx_u = rows[:, 0].astype(int)
    y_pseudo = np.array(rows[:, 2].tolist(), dtype=y_l.dtype)
    return anchor_pos, idx_u, y_pseudo


def _anchoral_select_anchors(
    Z_l: np.ndarray,
    y_l: np.ndarray,
    *,
    A_per_class: int,
    anchor_policy: str,
    anchor_total: int,
    anchor_alpha: float,
    anchor_steepness: float,
    random_state: int,
):
    rng = np.random.RandomState(random_state)
    y_l = np.asarray(y_l)
    anchor_pos = []

    anchor_counts = _compute_anchor_counts(
        y_l=y_l,
        policy=anchor_policy,
        total_anchors=anchor_total,
        A_per_class=A_per_class,
        anchor_alpha=anchor_alpha,
        anchor_steepness=anchor_steepness,
        random_state=random_state,
    )

    for cls in np.unique(y_l):
        n_a = int(anchor_counts.get(cls, 0))
        if n_a <= 0:
            continue

        mask = (y_l == cls)
        Zc = Z_l[mask]
        idx_c = np.where(mask)[0]
        if Zc.shape[0] == 0:
            continue

        if Zc.shape[0] <= n_a:
            anchor_pos.extend(idx_c.tolist())
        else:
            _, seeds = kmeans_plusplus(Zc, n_clusters=n_a, random_state=rng)
            anchor_pos.extend(idx_c[seeds].tolist())

    anchor_pos = np.asarray(anchor_pos, dtype=int)
    if anchor_pos.size == 0:
        # fallback: random labeled anchors
        take = min(max(1, A_per_class), len(Z_l))
        anchor_pos = rng.choice(len(Z_l), size=take, replace=False)

    return anchor_pos


def _anchoral_subpool_from_anchors(
    Z_l: np.ndarray,
    Z_u: np.ndarray,
    anchor_pos: np.ndarray,
    *,
    K: int,
    M: int,
    metric: str,
):
    """
    Build a subpool using a fixed set of anchor positions (no anchor selection).
    Returns:
      idx_u (indices into Z_u)
      dists (min distance per idx_u)
    """
    if anchor_pos is None or len(anchor_pos) == 0 or len(Z_u) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    nn = NearestNeighbors(metric=("cosine" if metric == "cosine" else "euclidean")).fit(Z_u)
    K_eff = min(int(K), len(Z_u))
    if K_eff <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    best_dist = {}
    for pos in anchor_pos:
        dists, idxs = nn.kneighbors([Z_l[pos]], n_neighbors=K_eff, return_distance=True)
        for d, i in zip(dists[0], idxs[0]):
            if i not in best_dist or d < best_dist[i]:
                best_dist[i] = d

    if len(best_dist) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    items = sorted(best_dist.items(), key=lambda kv: kv[1])
    if M is not None and int(M) > 0:
        items = items[: int(M)]

    idx_u = np.array([i for i, _ in items], dtype=int)
    dists = np.array([d for _, d in items], dtype=float)
    return idx_u, dists
def init_classifier(name, args, cfg):
    if name == 'MLP':
        return TorchMLPClassifier(cfg,
                                  hidden_layer_sizes=tuple([100]),
                                  max_iter=100,
                                  batch_size=64,
                                  lr=1e-3,
                                  random_state=args.random_state,
                                  device="cuda")
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100,
                                      random_state=args.random_state,
                                      n_jobs=-1)
    if name == 'XGBC':
        return XGBClassifier(use_label_encoder=False, learning_rate=0.1, max_depth=6, n_estimators=100,
                             eval_metric='mlogloss',
                             random_state=args.random_state)
    raise ValueError(name)


def init_generator(name, cfg):
    if name == 'TVAE':
        return TVAE(epochs=100, batch_size=60)
    if name == 'CTGAN':
        return CTGAN(epochs=100, batch_size=60)
    if name == 'RTF':
        return REaLTabFormer(model_type='tabular',
                             epochs=100,
                             gradient_accumulation_steps=1,
                             logging_steps=100,
                             numeric_max_len=12)
    raise ValueError(name)


def compute_metrics(y_pred, y_true):
    rpt = classification_report(y_true, y_pred, digits=4)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro')
    }, rpt


def _discrete_cols_with_label(cfg, df):
    cols = list(getattr(cfg.DATASET, "DISCRETE_FEATURES", []))
    if 'Label' in df.columns and 'Label' not in cols:
        cols = cols + ['Label']
    return cols


def _fit_generator(gen, df, discrete_cols):
    """
    Normalize generator.fit(...) calls across TVAE/CTGAN/RTF, since signatures can differ.
    """
    try:
        gen.fit(df, discrete_cols)
        return
    except TypeError:
        pass
    try:
        gen.fit(df, discrete_columns=discrete_cols)
        return
    except TypeError:
        pass
    gen.fit(df)


def _sample_generator(gen, args, n):
    if n <= 0:
        return pd.DataFrame()
    if args.generator == "RTF":
        return gen.sample(n_samples=n)
    try:
        return gen.sample(samples=n)
    except TypeError:
        return gen.sample(n)


def _sample_and_split_by_label(gen, args, fn, yal_dtype, k_per_cls, gen_train_df_len, random_state):
    """
    Group-training helper:
      - samples a big batch, then slices/pads per class to meet k_per_cls exactly.
      - if generator doesn't output Label, we assign labels in the exact needed proportions.
    """
    need_total = int(sum(int(v) for v in k_per_cls.values()))
    if need_total <= 0:
        return pd.DataFrame(columns=fn + ['Label'])

    # oversample to reduce risk of missing classes
    gen_total = max(need_total * 3, need_total)

    dfs_all = _sample_generator(gen, args, gen_total)
    if len(dfs_all) == 0:
        return pd.DataFrame(columns=fn + ['Label'])

    if 'Label' not in dfs_all.columns:
        labels = []
        for cls, k in k_per_cls.items():
            labels.extend([cls] * int(k))
        if len(labels) == 0:
            labels = [list(k_per_cls.keys())[0]] * len(dfs_all)
        if len(labels) < len(dfs_all):
            labels = (labels * int(math.ceil(len(dfs_all) / max(1, len(labels)))))[:len(dfs_all)]
        else:
            labels = labels[:len(dfs_all)]
        dfs_all['Label'] = np.array(labels, dtype=yal_dtype)

    try:
        dfs_all['Label'] = dfs_all['Label'].astype(yal_dtype, copy=False)
    except Exception:
        pass

    parts = []
    for cls, k in k_per_cls.items():
        k = int(k)
        if k <= 0:
            continue
        df_cls = dfs_all[dfs_all['Label'] == cls]
        if len(df_cls) >= k:
            parts.append(df_cls.iloc[:k])
        else:
            if len(df_cls) > 0:
                pad = df_cls.sample(n=(k - len(df_cls)), replace=True, random_state=random_state)
                parts.append(pd.concat([df_cls, pad], ignore_index=True))
            else:
                fallback = dfs_all.sample(n=k, replace=True, random_state=random_state).copy()
                fallback['Label'] = cls
                parts.append(fallback)

    return pd.concat(parts, ignore_index=True) if len(parts) else pd.DataFrame(columns=fn + ['Label'])


def knn_retrieve_neighbors(X_u, y_u, X_p, y_p, filter_bad_neighbors=False, n_neighbors=1):
    """
    Return (X_neighbors, y_pseudo) where y_pseudo is the assumed class label (anchor class).

    filter_bad_neighbors=True removes neighbors whose true y_p != assumed class.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_p)

    X_neighbors_all = []
    y_pseudo_all = []

    total_correct = 0
    total_neighbors = 0
    total_kept = 0

    for cls in np.unique(y_u):
        mask = (y_u == cls)
        Xu = X_u[mask]

        _, idxs = nn.kneighbors(Xu)
        idxs = idxs.reshape(-1)

        y_nn_true = y_p[idxs]
        correct_mask = (y_nn_true == cls)

        correct = int(np.sum(correct_mask))
        total = int(len(idxs))
        pct = (correct / total * 100.0) if total > 0 else 0.0

        if filter_bad_neighbors:
            idxs = idxs[correct_mask]

        kept = int(len(idxs))
        kept_pct = (kept / total * 100.0) if total > 0 else 0.0

        print(f"KNN neighbors | assumed class={cls}: {correct}/{total} match ({pct:.2f}%). "
              f"Kept={kept}/{total} ({kept_pct:.2f}%) | filter_bad_neighbors={filter_bad_neighbors}")

        total_correct += correct
        total_neighbors += total
        total_kept += kept

        if kept > 0:
            Xn = X_p[idxs]
            X_neighbors_all.append(Xn)
            y_pseudo_all.append(np.full(kept, cls, dtype=y_u.dtype))

    if len(X_neighbors_all) == 0:
        X_neighbors = np.empty((0, X_p.shape[1]))
        y_pseudo = np.empty((0,), dtype=y_u.dtype)
    else:
        X_neighbors = np.vstack(X_neighbors_all)
        y_pseudo = np.hstack(y_pseudo_all)

    overall_pct = (total_correct / total_neighbors * 100.0) if total_neighbors > 0 else 0.0
    overall_kept_pct = (total_kept / total_neighbors * 100.0) if total_neighbors > 0 else 0.0
    print(f"KNN neighbors | overall: {total_correct}/{total_neighbors} match ({overall_pct:.2f}%). "
          f"Kept={total_kept}/{total_neighbors} ({overall_kept_pct:.2f}%) | filter_bad_neighbors={filter_bad_neighbors}")

    return X_neighbors, y_pseudo


def base_set_up(args):
    cfg = get_config(args.dataset, args.al_method)
    Xtr, ytr, Xv, yv, Xt, yt = load_data(cfg)
    print(f"Initial set shapes | train: {Xtr.shape}, val: {Xv.shape}, test: {Xt.shape}")
    print_dist("original label", ytr)
    print_dist("original unlabel", yv)
    print_dist("original test", yt)
    if cfg.DATASET.STANDARDIZE:
        s = StandardScaler().fit(Xtr)
        Xtr, Xv, Xt = s.transform(Xtr), s.transform(Xv), s.transform(Xt)
    clf = init_classifier(args.classifier, args, cfg)
    clf.fit(Xtr, ytr)
    return cfg, clf, Xtr, ytr, Xv, yv, Xt, yt


def active_function(args, Xtr, Xv, ytr, yv, clf):
    sel = _init_al_method(args).sample(Xv, args.budget, clf, Xtr)
    u, c = np.unique(yv[sel], return_counts=True)
    print(u, c)
    if sel is not None and len(sel):
        Xtr = np.vstack([Xtr, Xv[sel]]); ytr = np.hstack([ytr, yv[sel]])
    return Xtr, ytr, sel


def active_learning_loops(args, Xtr, ytr, Xv, yv, clf, Xt=None, yt=None):
    """
    Run multiple active learning loops, removing selected points from the pool each step.
    Returns updated Xtr/ytr, remaining pool Xv/yv, and selected indices into the ORIGINAL pool.
    """
    idx_pool = np.arange(len(yv))
    all_sel = []
    step_pool_f1 = []
    step_test_f1 = []

    for step in range(max(0, int(args.al_steps))):
        if len(Xv) == 0:
            print(f"[AL] step {step+1}: pool empty, stopping.")
            break

        # Train on current labeled set to select next samples.
        clf.fit(Xtr, ytr)
        sel = _init_al_method(args).sample(Xv, args.budget, clf, Xtr)

        if sel is None or len(sel) == 0:
            print(f"[AL] step {step+1}: no selection, stopping.")
            break

        u, c = np.unique(yv[sel], return_counts=True)
        print(f"[AL] step {step+1}/{args.al_steps} selected counts:", dict(zip(u, c)))

        Xtr = np.vstack([Xtr, Xv[sel]])
        ytr = np.hstack([ytr, yv[sel]])
        all_sel.append(idx_pool[sel])

        mask = np.ones(len(Xv), dtype=bool)
        mask[sel] = False
        Xv = Xv[mask]
        yv = yv[mask]
        idx_pool = idx_pool[mask]

        # Retrain after adding new labeled data and evaluate.
        clf.fit(Xtr, ytr)
        if len(Xv) > 0:
            yv_pred = clf.predict(Xv)
            f1_macro = f1_score(yv, yv_pred, average='macro', zero_division=0)
            print(f"[AL] step {step+1}: pool macro F1={f1_macro:.4f}")
            step_pool_f1.append(float(f1_macro))
        if Xt is not None and yt is not None:
            yt_pred = clf.predict(Xt)
            f1_macro = f1_score(yt, yt_pred, average='macro', zero_division=0)
            print(f"[AL] step {step+1}: test macro F1={f1_macro:.4f}")
            step_test_f1.append(float(f1_macro))

    if len(all_sel):
        sel_all = np.hstack(all_sel).astype(int)
    else:
        sel_all = np.array([], dtype=int)
    return Xtr, ytr, Xv, yv, sel_all, step_pool_f1, step_test_f1



def run_base(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr0, ytr0 = Xtr.copy(), ytr.copy()
    Xv0, yv0 = Xv.copy(), yv.copy()
    Xt0, yt0 = Xt.copy(), yt.copy()
    Xtr, ytr, Xv, yv, _, step_pool_f1, step_test_f1 = active_learning_loops(
        args, Xtr, ytr, Xv, yv, clf, Xt, yt
    )
    export_dataset_versions(
        args, cfg, results_dir, Xtr0, ytr0, Xv0, yv0, Xt0, yt0, Xal=Xtr, yal=ytr
    )
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xt)
    return y_pred, yt, {"al_pool_f1": step_pool_f1, "al_test_f1": step_test_f1, "densegaal_test_f1": []}


def run_base_pooling(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr0, ytr0 = Xtr.copy(), ytr.copy()
    Xv0, yv0 = Xv.copy(), yv.copy()
    Xt0, yt0 = Xt.copy(), yt.copy()
    Xv, yv = init_pooling(args.pooling_method, Xtr, ytr, Xv, yv)
    Xtr, ytr, Xv, yv, _, step_pool_f1, step_test_f1 = active_learning_loops(
        args, Xtr, ytr, Xv, yv, clf, Xt, yt
    )
    export_dataset_versions(
        args, cfg, results_dir, Xtr0, ytr0, Xv0, yv0, Xt0, yt0, Xal=Xtr, yal=ytr
    )
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xt)
    return y_pred, yt, {"al_pool_f1": step_pool_f1, "al_test_f1": step_test_f1, "densegaal_test_f1": []}


def run_augmented(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr0, ytr0 = Xtr.copy(), ytr.copy()
    Xv0, yv0 = Xv.copy(), yv.copy()
    Xt0, yt0 = Xt.copy(), yt.copy()
    Xal, yal, Xv, yv, _, step_pool_f1, step_test_f1 = active_learning_loops(
        args, Xtr, ytr, Xv, yv, clf, Xt, yt
    )
    # Train classifier AFTER AL selections for synthetic filtering.
    clf.fit(Xal, yal)
    filter_clf = clf

    # parse new multipliers
    multipliers = _parse_num_synthetic_list(args.num_synthetic)

    u, c = np.unique(yal, return_counts=True)
    maj = u[np.argmax(c)]
    fn = cfg.DATASET.FEATURE_NAMES

    # Augmentable classes are ALL classes (including majority)
    underrep_classes = [cls for cls in u]
    if len(underrep_classes) == 0:
        print("[WARN] Only one class present. Skipping augmentation.")
        Xf, yf = Xal, yal
        export_dataset_versions(
            args, cfg, results_dir, Xtr0, ytr0, Xv0, yv0, Xt0, yt0, Xal=Xal, yal=yal, Xaug=Xf, yaug=yf
        )
        clf.fit(Xal, yal)
        y_pred = clf.predict(Xt)
        return y_pred, yt, {"al_pool_f1": step_pool_f1, "al_test_f1": step_test_f1, "densegaal_test_f1": []}

    # Compute per-class synthetic counts (NEW DEFAULT: compute_synthetic_count everywhere)
    k_per_cls = compute_k_per_cls(
        y_labeled=yal,
        classes_to_aug=underrep_classes,
        alpha=args.alpha,
        steepness=args.steepness,
        decay_power=args.decay_power,
        multipliers=multipliers,
    )

    print("DA k_per_cls (class -> synthetic count):", k_per_cls)

    # Train one generator on underrep combined (same as before), but sample per-class targets
    dfu = pd.DataFrame(Xal, columns=fn)
    dfu['Label'] = yal
    gen = init_generator(args.generator, cfg)
    disc = _discrete_cols_with_label(cfg, dfu)
    _fit_generator(gen, dfu, disc)

    comb = _sample_and_split_by_label(
        gen=gen,
        args=args,
        fn=fn,
        yal_dtype=yal.dtype,
        k_per_cls=k_per_cls,
        gen_train_df_len=len(dfu),
        random_state=args.random_state
    )

    if len(comb):
        print_dist('Synthetic', comb['Label'].values)
        Xs, ys = comb[fn].values, comb['Label'].values
    else:
        Xs = np.empty((0, Xal.shape[1]))
        ys = np.empty((0,), dtype=yal.dtype)

    if args.filter_synthetic and len(ys):
        msk = filter_clf.predict(Xs) != 0
        Xs, ys = Xs[msk], ys[msk]

    Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
    yf = np.hstack([yal, ys]) if len(ys) else yal

    export_dataset_versions(
        args, cfg, results_dir, Xtr0, ytr0, Xv0, yv0, Xt0, yt0, Xal=Xal, yal=yal, Xaug=Xf, yaug=yf
    )
    clf.fit(Xf, yf)
    y_pred = clf.predict(Xt)
    return y_pred, yt, {"al_pool_f1": step_pool_f1, "al_test_f1": step_test_f1, "densegaal_test_f1": []}


def _run_densegaal_with_state(args, cfg, clf, Xal, yal, Xv, yv, Xt, yt, Xv0, yv0, _sel):
    # Train classifier AFTER AL selections for synthetic filtering.
    print(Xal.shape)
    clf.fit(Xal, yal)
    filter_clf = clf

    # parse new multipliers
    multipliers = _parse_num_synthetic_list(args.num_synthetic)

    fn = cfg.DATASET.FEATURE_NAMES
    labeled_df = pd.DataFrame(Xal, columns=fn)
    labeled_df['Label'] = yal

    u, c = np.unique(yal, return_counts=True)
    freqs = c / len(yal) if len(yal) else np.array([])

    worst_class = None

    # ------------------------------------------------------------------
    # ORIGINAL DenseGAAL MODE: anchor selection + kNN
    # ------------------------------------------------------------------
    if args.densegaal_anchoral:
        print("DenseGAAL: densegaal_anchoral mode enabled (kmeans++ anchors + ranked top-M subpool).")
        _m_str = 'ALL' if (args.anchoral_M is None or args.anchoral_M <= 0) else str(args.anchoral_M)
        print(f"  rep={args.rep} metric={args.metric} A_per_class={args.anchoral_A} K={args.anchoral_K} M={_m_str}")

        # build unlabeled pool (AL-selected points already removed)
        X_pool = Xv
        y_pool_true = yv  # available in simulation; NOT used for pseudo labeling

        if len(X_pool) == 0:
            print("[WARN] Unlabeled pool empty after masking; skipping subpooling.")
            clf.fit(Xal, yal)
            y_pred = clf.predict(Xt)
            return y_pred, yt, Xal, yal

        enc = _make_encoder(args, cfg)
        enc.fit(Xal, yal, X_pool, cfg)

        Z_l = enc.transform(Xal)
        Z_u = enc.transform(X_pool)

        if args.no_local_support:
            print("DenseGAAL: no_local_support enabled (skipping kNN; generator uses anchors only).")
            anchor_pos = _anchoral_select_anchors(
                Z_l=Z_l,
                y_l=yal,
                A_per_class=args.anchoral_A,
                anchor_policy='per_class',
                anchor_total=args.anchoral_anchor_total,
                anchor_alpha=args.alpha,
                anchor_steepness=args.steepness,
                random_state=args.random_state,
            )
            idx_u = np.array([], dtype=int)
            y_pseudo = np.array([], dtype=yal.dtype)
        else:
            anchor_pos, idx_u, y_pseudo = _anchoral_subpool_with_pseudolabels(
                Z_l=Z_l,
                y_l=yal,
                Z_u=Z_u,
                A_per_class=args.anchoral_A,
                anchor_policy='per_class',
                anchor_total=args.anchoral_anchor_total,
                K=args.anchoral_K,
                M=args.anchoral_M,
                metric=args.metric,
                anchor_alpha=args.alpha,
                anchor_steepness=args.steepness,
                random_state=args.random_state,
            )

        _anchor_labels = yal[anchor_pos]
        _ctr = Counter(_anchor_labels.tolist())
        _total = len(anchor_pos)
        _props = {int(k): round(v / _total, 3) for k, v in _ctr.items()} if _total else {}
        print(f"[anchors] total_selected={_total}")
        print(f"[anchors] per-class counts: {dict(_ctr)}")
        print(f"[anchors] per-class proportions: {_props}")

        # subpool data (pseudo-labeled)
        X_sub = X_pool[idx_u]
        y_sub = y_pseudo
        print_dist("DenseGAAL-subpool (pseudo-labeled)", y_sub)

        if args.neighbor_only:
            if args.no_local_support:
                print("[WARN] neighbor_only + no_local_support: no neighbors to add; returning labeled pool unchanged.")
                Xf = Xal
                yf = yal
                clf.fit(Xf, yf)
                y_pred = clf.predict(Xt)
                return y_pred, yt, Xf, yf
            print("DenseGAAL: neighbor-only mode enabled (adding subpool directly; skipping generator).")
            Xf = np.vstack([Xal, X_sub]) if len(X_sub) else Xal
            yf = np.hstack([yal, y_sub]) if len(y_sub) else yal
            clf.fit(Xf, yf)
            y_pred = clf.predict(Xt)
            return y_pred, yt, Xf, yf

        # generator training dataset:
        #   - base: anchors only (as in original DenseGAAL intent)
        #   - optional: all labeled so far
        anchors_df = pd.DataFrame(Xal[anchor_pos], columns=fn)
        anchors_df["Label"] = yal[anchor_pos]

        sub_df = pd.DataFrame(X_sub, columns=fn)
        sub_df["Label"] = y_sub

        base_gen_df = labeled_df if args.gen_train_all_labeled else anchors_df
        gen_train_df = pd.concat([base_gen_df, sub_df], ignore_index=True)

        # augment these classes (only those represented by selected anchors)
        classes_to_aug = np.unique(anchors_df["Label"].values)

        k_per_cls = compute_k_per_cls(
            y_labeled=yal,
            classes_to_aug=classes_to_aug,
            alpha=args.alpha,
            steepness=args.steepness,
            decay_power=args.decay_power,
            multipliers=multipliers,
        )

        print("DenseGAAL(densegaal_anchoral) k_per_cls (class -> synthetic count):", k_per_cls)

        syn = []
        for cls in classes_to_aug:
            if args.gen_train_all_labeled:
                base_cls = labeled_df[labeled_df["Label"] == cls]
            else:
                base_cls = anchors_df[anchors_df["Label"] == cls]
                if len(base_cls) == 0:
                    base_cls = labeled_df[labeled_df["Label"] == cls]
            sub_cls = sub_df[sub_df["Label"] == cls]

            train_df_cls = pd.concat([base_cls, sub_cls], ignore_index=True)
            if len(train_df_cls) < 2:
                print(f"[WARN] Skipping generator for class={cls} (train_df_cls has {len(train_df_cls)} rows).")
                continue

            gen = init_generator(args.generator, cfg)
            disc = _discrete_cols_with_label(cfg, train_df_cls)
            _fit_generator(gen, train_df_cls, disc)

            k = int(k_per_cls.get(cls, 0))
            dfs = _sample_generator(gen, args, k)
            if len(dfs) == 0:
                continue
            dfs["Label"] = cls
            syn.append(dfs)

        comb = pd.concat(syn, ignore_index=True) if len(syn) else pd.DataFrame(columns=fn + ["Label"])

        if len(comb):
            Xs = comb[fn].values
            ys = comb["Label"].values
        else:
            Xs = np.empty((0, Xal.shape[1]))
            ys = np.empty((0,), dtype=yal.dtype)

        if args.filter_synthetic and len(ys):
            msk = filter_clf.predict(Xs) != 0
            Xs, ys = Xs[msk], ys[msk]

        Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
        yf = np.hstack([yal, ys]) if len(ys) else yal

        clf.fit(Xf, yf)
        y_pred = clf.predict(Xt)
        return y_pred, yt, Xf, yf

    # ------------------------------------------------------------------
    # ORIGINAL DenseGAAL MODE: anchor selection + kNN
    # ------------------------------------------------------------------
    rng = np.random.RandomState(args.random_state)
    Z_l = None

    def _pick_anchors_for_class(cls_idx, n_cls):
        if n_cls <= 0 or len(cls_idx) == 0:
            return []
        if False:  # kmeans++ anchor selection removed
            Zc = Z_l[cls_idx]
            if Zc.shape[0] <= n_cls:
                return cls_idx.tolist()
            _, seeds = kmeans_plusplus(Zc, n_clusters=n_cls, random_state=rng)
            return cls_idx[seeds].tolist()
        # default random
        return rng.choice(cls_idx, size=n_cls, replace=False).tolist()

    all_choices = []
    if args.minority:
        # minority uses TWO most underrepresented classes
        sorted_classes_index = np.argsort(freqs)[:2]
        minority_classes = u[sorted_classes_index]
        print(f"using minority class densegaal composition (2 classes): {minority_classes}")

        for cls in minority_classes:
            cls_idx = np.where(yal == cls)[0]
            cls_freq = freqs[u.tolist().index(cls)]
            frac = compute_anchor_fraction(cls_freq, alpha=args.alpha, steepness=args.steepness)
            n_cls = max(1, int(frac * len(cls_idx)))
            chosen = _pick_anchors_for_class(cls_idx, n_cls)
            print(len(cls_idx) * frac)
            all_choices.extend(chosen)
    else:
        for cls in u:
            cls_idx = np.where(yal == cls)[0]
            cls_freq = freqs[u.tolist().index(cls)]
            frac = compute_anchor_fraction(cls_freq, alpha=args.alpha, steepness=args.steepness)
            print(f"class={cls} freq={cls_freq:.4f} frac={frac:.4f}")
            n_cls = max(1, int(frac * len(cls_idx)))
            chosen = _pick_anchors_for_class(cls_idx, n_cls)
            all_choices.extend(chosen)

    choice = np.array(all_choices)
    Xu, yu = Xal[choice], yal[choice]

    print_dist('Anchors', yu)

    # pool already excludes AL-selected points after active_learning_loops
    X_pool = Xv
    y_pool = yv
    # retrieve neighbors ONCE (used for neighbor-only and generator training)
    if args.no_local_support:
        print("DenseGAAL: no_local_support enabled (skipping kNN; generator uses anchors only).")
        Xn = np.empty((0, X_pool.shape[1]))
        yn = np.empty((0,), dtype=yal.dtype)
    else:
        Xn, yn = knn_retrieve_neighbors(
            Xu, yu, X_pool, y_pool,
            filter_bad_neighbors=args.filter_bad_neighbors,
            n_neighbors=args.alfa_k
        )
        print_dist("Neighbors (pseudo-labeled)", yn)

    if args.neighbor_only:
        if args.no_local_support:
            print("[WARN] neighbor_only + no_local_support: no neighbors to add; returning labeled pool unchanged.")
            Xf = Xal
            yf = yal
            clf.fit(Xf, yf)
            y_pred = clf.predict(Xt)
            return y_pred, yt, Xf, yf
        print("DenseGAAL: neighbor-only mode enabled (skipping generator + synthetic data).")
        Xf = np.vstack([Xal, Xn]) if len(Xn) else Xal
        yf = np.hstack([yal, yn]) if len(yn) else yal
        clf.fit(Xf, yf)
        y_pred = clf.predict(Xt)
        return y_pred, yt, Xf, yf

    anchors_df = pd.DataFrame(Xu, columns=fn)
    anchors_df['Label'] = yu

    neighbors_df = pd.DataFrame(Xn, columns=fn) if len(Xn) else pd.DataFrame(columns=fn)
    if len(Xn):
        neighbors_df['Label'] = yn
    else:
        neighbors_df['Label'] = pd.Series([], dtype=yal.dtype)

    base_gen_df = labeled_df if args.gen_train_all_labeled else anchors_df
    gen_train_df = pd.concat([base_gen_df, neighbors_df], ignore_index=True)

    # We will augment these classes (only those represented by selected anchors)
    classes_to_aug = np.unique(anchors_df["Label"].values)

    k_per_cls = compute_k_per_cls(
        y_labeled=yal,
        classes_to_aug=classes_to_aug,
        alpha=args.alpha,
        steepness=args.steepness,
        decay_power=args.decay_power,
        multipliers=multipliers,
    )

    print("DenseGAAL k_per_cls (class -> synthetic count):", k_per_cls)

    syn = []

    for cls in classes_to_aug:
            k = int(k_per_cls.get(cls, 0))
            if k == 0:
                continue
            if args.gen_train_all_labeled:
                base_cls = labeled_df[labeled_df['Label'] == cls]
            else:
                base_cls = anchors_df[anchors_df['Label'] == cls]
                if len(base_cls) == 0:
                    # Fallback: if no anchors for this class, use all labeled samples of the class
                    base_cls = labeled_df[labeled_df['Label'] == cls]
            neigh_cls = neighbors_df[neighbors_df['Label'] == cls] if len(neighbors_df) else neighbors_df.iloc[0:0]

            train_df_cls = pd.concat([base_cls, neigh_cls], ignore_index=True)

            if len(train_df_cls) < 2:
                print(f"[WARN] Skipping generator for class={cls} (train_df_cls has {len(train_df_cls)} rows).")
                continue

            gen = init_generator(args.generator, cfg)
            disc = _discrete_cols_with_label(cfg, train_df_cls)
            _fit_generator(gen, train_df_cls, disc)

            print("generating", k, "samples for class", cls)
            dfs = _sample_generator(gen, args, k)
            if len(dfs) == 0:
                continue

            dfs['Label'] = cls
            syn.append(dfs)

    comb = pd.concat(syn, ignore_index=True) if len(syn) else pd.DataFrame(columns=fn + ['Label'])

    if len(comb):
        Xs = comb[fn].values
        ys = comb['Label'].values
    else:
        Xs = np.empty((0, Xal.shape[1]))
        ys = np.empty((0,), dtype=yal.dtype)

    if args.filter_synthetic and len(ys):
        msk = filter_clf.predict(Xs) != 0
        Xs, ys = Xs[msk], ys[msk]

    Xf = np.vstack([Xal, Xs]) if len(ys) else Xal
    yf = np.hstack([yal, ys]) if len(ys) else yal

    clf.fit(Xf, yf)
    y_pred = clf.predict(Xt)
    return y_pred, yt, Xf, yf


def run_densegaal(args, results_dir):
    cfg, clf, Xtr, ytr, Xv, yv, Xt, yt = base_set_up(args)
    Xtr0, ytr0 = Xtr.copy(), ytr.copy()
    Xv0_base, yv0_base = Xv.copy(), yv.copy()
    Xt0, yt0 = Xt.copy(), yt.copy()
    if args.pooling_method:
        print("using pooling")
        Xv, yv = init_pooling(args.pooling_method, Xtr, ytr, Xv, yv)

    al_pool_f1 = []
    al_test_f1 = []
    densegaal_test_f1 = []

    if int(getattr(args, 'densegaal_steps', 1)) > 1:
        if getattr(args, 'densegaal_steps_mode', 'per_step') == 'per_step':
            if int(getattr(args, 'al_steps', 1)) != 1:
                print("[WARN] densegaal_steps>1: running exactly 1 AL step per DenseGAAL iteration (ignoring al_steps>1).")
            Xal, yal = Xtr, ytr
            orig_al_steps = args.al_steps
            try:
                last_Xf, last_yf = None, None
                for step in range(int(args.densegaal_steps)):
                    Xv0, yv0 = Xv.copy(), yv.copy()
                    args.al_steps = 1
                    Xal, yal, Xv, yv, _sel, step_pool_f1, step_test_f1 = active_learning_loops(
                        args, Xal, yal, Xv, yv, clf, Xt, yt
                    )
                    al_pool_f1.extend(step_pool_f1)
                    al_test_f1.extend(step_test_f1)
                    y_pred, y_true, last_Xf, last_yf = _run_densegaal_with_state(
                        args, cfg, clf, Xal, yal, Xv, yv, Xt, yt, Xv0, yv0, _sel
                    )
                    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    densegaal_test_f1.append(float(f1_macro))
            finally:
                args.al_steps = orig_al_steps
            export_dataset_versions(
                args, cfg, results_dir, Xtr0, ytr0, Xv0_base, yv0_base, Xt0, yt0, Xal=Xal, yal=yal, Xaug=last_Xf, yaug=last_yf
            )
            return y_pred, y_true, {"al_pool_f1": al_pool_f1, "al_test_f1": al_test_f1, "densegaal_test_f1": densegaal_test_f1}
        else:
            # per_al_steps: run ALL AL steps first (densegaal_steps times), then generate ONCE.
            Xal, yal = Xtr, ytr
            orig_al_steps = args.al_steps
            try:
                args.al_steps = int(args.densegaal_steps)
                Xv0, yv0 = Xv.copy(), yv.copy()
                Xal, yal, Xv, yv, _sel, step_pool_f1, step_test_f1 = active_learning_loops(
                    args, Xal, yal, Xv, yv, clf, Xt, yt
                )
                al_pool_f1.extend(step_pool_f1)
                al_test_f1.extend(step_test_f1)
                y_pred, y_true, last_Xf, last_yf = _run_densegaal_with_state(
                    args, cfg, clf, Xal, yal, Xv, yv, Xt, yt, Xv0, yv0, _sel
                )
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                densegaal_test_f1.append(float(f1_macro))
            finally:
                args.al_steps = orig_al_steps
            export_dataset_versions(
                args, cfg, results_dir, Xtr0, ytr0, Xv0_base, yv0_base, Xt0, yt0, Xal=Xal, yal=yal, Xaug=last_Xf, yaug=last_yf
            )
            return y_pred, y_true, {"al_pool_f1": al_pool_f1, "al_test_f1": al_test_f1, "densegaal_test_f1": densegaal_test_f1}

    # Single-iteration (default) behavior
    Xv0, yv0 = Xv.copy(), yv.copy()
    Xal, yal, Xv, yv, _sel, step_pool_f1, step_test_f1 = active_learning_loops(
        args, Xtr, ytr, Xv, yv, clf, Xt, yt
    )
    al_pool_f1.extend(step_pool_f1)
    al_test_f1.extend(step_test_f1)
    y_pred, y_true, last_Xf, last_yf = _run_densegaal_with_state(
        args, cfg, clf, Xal, yal, Xv, yv, Xt, yt, Xv0, yv0, _sel
    )
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    densegaal_test_f1.append(float(f1_macro))
    export_dataset_versions(
        args, cfg, results_dir, Xtr0, ytr0, Xv0_base, yv0_base, Xt0, yt0, Xal=Xal, yal=yal, Xaug=last_Xf, yaug=last_yf
    )
    return y_pred, y_true, {"al_pool_f1": al_pool_f1, "al_test_f1": al_test_f1, "densegaal_test_f1": densegaal_test_f1}


def main():
    args = parse_args()
    res = ensure_results_dir(args)
    print("Results dir:", res)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    if args.al_method == 'base' and not args.pooling_method:
        y_pred, y_true, step_metrics = run_base(args, res)
    elif args.al_method == 'DA':
        y_pred, y_true, step_metrics = run_augmented(args, res)
    elif args.pooling_method and args.al_method == "base":
        y_pred, y_true, step_metrics = run_base_pooling(args, res)
    else:
        y_pred, y_true, step_metrics = run_densegaal(args, res)
    elapsed_sec = time.perf_counter() - start_time
    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    m, rpt = compute_metrics(y_pred, y_true)
    with open(os.path.join(res, 'report.txt'), 'w') as f:
        f.write(rpt)
        if step_metrics:
            al_pool_f1 = step_metrics.get("al_pool_f1", [])
            al_test_f1 = step_metrics.get("al_test_f1", [])
            densegaal_test_f1 = step_metrics.get("densegaal_test_f1", [])
            if al_pool_f1 or al_test_f1 or densegaal_test_f1:
                f.write("\n\n[Step metrics]\n")
            if al_pool_f1:
                f.write("al_pool_macro_f1:\n")
                for i, v in enumerate(al_pool_f1, 1):
                    f.write(f"  step_{i}: {v:.4f}\n")
            if al_test_f1:
                f.write("al_test_macro_f1:\n")
                for i, v in enumerate(al_test_f1, 1):
                    f.write(f"  step_{i}: {v:.4f}\n")
            if densegaal_test_f1:
                f.write("densegaal_test_macro_f1:\n")
                for i, v in enumerate(densegaal_test_f1, 1):
                    f.write(f"  step_{i}: {v:.4f}\n")
        neighbors_k = args.anchoral_K if getattr(args, 'densegaal_anchoral', False) else args.alfa_k
        multipliers_str = ",".join([str(x) for x in _parse_num_synthetic_list(args.num_synthetic)])
        f.write(
            "\n\n[Run metadata]\n"
            f"elapsed_seconds: {elapsed_sec:.4f}\n"
            f"random_state: {args.random_state}\n"
            f"decay_power: {args.decay_power}\n"
            f"steepness: {args.steepness}\n"
            f"neighbors_k: {neighbors_k}\n"
            f"num_synthetic_multipliers: {multipliers_str}\n"
            f"peak_gpu_mem_mb: {('n/a' if peak_gpu_mb is None else f'{peak_gpu_mb:.2f}')}\n"
        )
    with open(os.path.join(res, 'metrics.json'), 'w') as f:
        json.dump(m, f, indent=4)


if __name__ == '__main__':
    main()
