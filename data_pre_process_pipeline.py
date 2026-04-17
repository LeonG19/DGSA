#!/usr/bin/env python3
"""
split_and_save_npz.py

Creates train/val/test NPZ splits for AL experiments.

DEFAULT (no shift):
  - train: 50% random
  - val:   25% random
  - test:  25% random

CIC-IDS-LIKE SHIFT (domain shift):
  - train is sampled mostly from Domain A (labeled distribution)
  - val + test are sampled mostly from Domain B (unlabeled/test distribution)
  - then Domain B is split into val and test

Shift modes:
  --shift_mode none           : random split (no shift)
  --shift_mode cluster        : subpopulation drift via KMeans clusters
  --shift_mode quantile       : covariate drift via feature quantiles
  --shift_mode class_imbalance : label-distribution drift (Domain A majority-heavy, Domain B minority-heavy)

Also computes drift metrics between train and test:
  - PSI (per-feature + aggregate)
  - JS divergence (per-feature + aggregate)
  - MMD-RBF (global, multivariate)

Outputs:
  data/<output_dir>/train.npz, val.npz, test.npz
  data/<output_dir>/label2id.json
  data/<output_dir>/drift_report.json
  data/<output_dir>/drift_features.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from yacs.config import CfgNode as CN
from config import get_dataset_config
from ucimlrepo import fetch_ucirepo


# ----------------------------
# Discrete transform (your code)
# ----------------------------

def transform_discrete(df: pd.DataFrame, discrete_cols: list) -> pd.DataFrame:
    df_transformed = df.copy()
    le = LabelEncoder()
    for col in discrete_cols:
        if col in df_transformed.columns:
            df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
    return df_transformed


# ----------------------------
# Drop classes helper (NEW)
# ----------------------------

def parse_drop_classes_arg(arg: str) -> List[str]:
    """
    Supports:
      --drop_classes "a,b,c"
      --drop_classes "a, b , c"
      --drop_classes ""  (no-op)
    """
    if arg is None:
        return []
    s = str(arg).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def drop_classes_from_df(df: pd.DataFrame, label_col: str, drop_classes: List[str], tag: str = "df") -> pd.DataFrame:
    """
    Removes rows whose label is in drop_classes (string-matched after .astype(str)).
    """
    if not drop_classes:
        return df
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found while dropping classes. Columns={df.columns.tolist()}")

    y = df[label_col].astype(str)
    drop_set = set([str(c) for c in drop_classes])

    before = len(df)
    mask_keep = ~y.isin(drop_set)
    df_out = df.loc[mask_keep].copy()
    after = len(df_out)

    # Print a small summary (counts removed by class)
    removed = before - after
    if removed > 0:
        removed_counts = y.loc[~mask_keep].value_counts().to_dict()
        print(f"[drop_classes] {tag}: removed {removed}/{before} rows. Removed counts={removed_counts}")
    else:
        print(f"[drop_classes] {tag}: removed 0/{before} rows (no matches).")

    # Basic sanity: need at least 1 class left
    remaining_classes = df_out[label_col].astype(str).unique().tolist()
    if len(remaining_classes) == 0:
        raise RuntimeError(
            f"After dropping classes {sorted(list(drop_set))}, no data remains in {tag}."
        )
    if len(remaining_classes) < 2:
        print(
            f"[drop_classes] WARNING: only {len(remaining_classes)} class remains in {tag}: {remaining_classes}. "
            f"Some shift modes and/or stratified operations may be less meaningful."
        )
    return df_out


# ----------------------------
# Preprocess features (FIX: keep rows aligned with labels)
# ----------------------------

def preprocess_aligned(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List[str]]:
    """
    Returns:
      df_clean (DataFrame): cleaned + aligned + reset_index
      X (ndarray): feature matrix aligned with df_clean
      y (Series): labels aligned with df_clean (index matches df_clean)
      feature_names (list): feature column names
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame columns: {df.columns.tolist()}")

    df_work = df.copy()

    # Ensure index is sane and unique for consistent positional indexing
    df_work = df_work.reset_index(drop=True)

    y = df_work[label_col].astype(str)
    Xdf = df_work.drop(columns=[label_col]).copy()
    feature_names = list(Xdf.columns)

    Xdf = Xdf.apply(pd.to_numeric, errors='coerce')
    Xdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xdf.fillna(Xdf.mean(numeric_only=True), inplace=True)

    # Drop remaining NaNs consistently
    mask = ~Xdf.isna().any(axis=1)
    Xdf = Xdf.loc[mask].copy()
    y = y.loc[mask].copy()

    # Build cleaned df with aligned rows only, and reset index again
    df_clean = pd.concat([Xdf, y.rename(label_col)], axis=1).reset_index(drop=True)
    X = df_clean.drop(columns=[label_col]).values
    y = df_clean[label_col].astype(str)

    return df_clean, X, y, feature_names


# ----------------------------
# Drift metrics
# ----------------------------

def _safe_hist_prob(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bins)
    p = counts.astype(float)
    p = p / max(p.sum(), 1.0)
    eps = 1e-12
    return np.clip(p, eps, 1.0)

def psi_feature(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 5 or len(test) < 5:
        return 0.0

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train, qs))
    if len(edges) < 3:
        edges = np.linspace(np.min(train), np.max(train) + 1e-9, n_bins + 1)

    p = _safe_hist_prob(train, edges)
    q = _safe_hist_prob(test, edges)
    return float(np.sum((q - p) * np.log(q / p)))

def js_divergence_1d(train: np.ndarray, test: np.ndarray, n_bins: int = 30) -> float:
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 5 or len(test) < 5:
        return 0.0

    lo = min(np.min(train), np.min(test))
    hi = max(np.max(train), np.max(test))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    p = _safe_hist_prob(train, edges)
    q = _safe_hist_prob(test, edges)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None, max_points: int = 2000) -> float:
    rng = np.random.default_rng(42)

    if X.shape[0] > max_points:
        X = X[rng.choice(X.shape[0], max_points, replace=False)]
    if Y.shape[0] > max_points:
        Y = Y[rng.choice(Y.shape[0], max_points, replace=False)]

    Z = np.vstack([X, Y])

    if gamma is None:
        idx = rng.choice(Z.shape[0], min(800, Z.shape[0]), replace=False)
        Zs = Z[idx]
        dists = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=2)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d2)

    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    m = X.shape[0]
    n = Y.shape[0]
    if m < 2 or n < 2:
        return 0.0

    mmd2 = (Kxx.sum() / (m * (m - 1))) + (Kyy.sum() / (n * (n - 1))) - (2.0 * Kxy.mean())
    return float(np.sqrt(max(mmd2, 0.0)))

def compute_drift_report(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    psi_bins: int = 10,
    js_bins: int = 30,
    compute_mmd_flag: bool = True
) -> Tuple[Dict[str, float], pd.DataFrame]:
    psis, jss = [], []
    for j in range(X_train.shape[1]):
        psis.append(psi_feature(X_train[:, j], X_test[:, j], n_bins=psi_bins))
        jss.append(js_divergence_1d(X_train[:, j], X_test[:, j], n_bins=js_bins))

    feat_df = pd.DataFrame({
        "feature": feature_names[:X_train.shape[1]],
        "psi": psis,
        "js_divergence": jss
    }).sort_values("psi", ascending=False).reset_index(drop=True)

    report = {
        "psi_mean": float(np.mean(psis)) if len(psis) else 0.0,
        "psi_median": float(np.median(psis)) if len(psis) else 0.0,
        "psi_max": float(np.max(psis)) if len(psis) else 0.0,
        "js_mean": float(np.mean(jss)) if len(jss) else 0.0,
        "js_median": float(np.median(jss)) if len(jss) else 0.0,
        "js_max": float(np.max(jss)) if len(jss) else 0.0,
    }
    if compute_mmd_flag:
        report["mmd_rbf"] = mmd_rbf(X_train, X_test)
    return report, feat_df


# ----------------------------
# Sampling helper (preserve label proportions roughly)
# ----------------------------

def _stratified_sample_indices(
    y_int: np.ndarray,
    candidate_idx: np.ndarray,
    n_total: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample n_total indices from candidate_idx approximately preserving the class distribution
    found within candidate_idx.
    """
    if n_total <= 0 or len(candidate_idx) == 0:
        return np.array([], dtype=int)

    y_c = y_int[candidate_idx]
    classes, counts = np.unique(y_c, return_counts=True)
    probs = counts / counts.sum()

    chosen = []
    for c, p in zip(classes, probs):
        want = int(round(n_total * float(p)))
        c_idx = candidate_idx[y_c == c]
        if want <= 0 or len(c_idx) == 0:
            continue
        take = min(want, len(c_idx))
        chosen.extend(rng.choice(c_idx, take, replace=False).tolist())

    if len(chosen) < n_total:
        remaining = np.setdiff1d(candidate_idx, np.array(chosen, dtype=int), assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, min(n_total - len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())

    if len(chosen) > n_total:
        chosen = rng.choice(np.array(chosen), n_total, replace=False).tolist()

    return np.array(chosen, dtype=int)


# ----------------------------
# CIC-like Domain Shift Splitters
# ----------------------------

def split_domain_shift_cluster(
    df: pd.DataFrame,
    label_col: str,
    drift_strength: float,
    n_clusters: int,
    pca_dim: Optional[int],
    random_state: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    df_clean, X_all, y_all_str, _ = preprocess_aligned(df, label_col=label_col)
    y_all_int = LabelEncoder().fit_transform(y_all_str.to_numpy())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_all)

    if pca_dim is not None and pca_dim > 0 and pca_dim < Xs.shape[1]:
        Z = PCA(n_components=pca_dim, random_state=random_state).fit_transform(Xs)
    else:
        Z = Xs

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(Z)

    idx_all = np.arange(len(df_clean))

    all_c = np.unique(clusters)
    rng.shuffle(all_c)
    half = max(1, len(all_c) // 2)
    domainA = set(all_c[:half].tolist())
    domainB = set(all_c[half:].tolist()) if len(all_c) > 1 else set(all_c.tolist())

    idx_A = idx_all[np.isin(clusters, list(domainA))]
    idx_B = idx_all[np.isin(clusters, list(domainB))]

    n_total = len(idx_all)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, n_total - n_train - n_test)

    n_train_A = int(round(n_train * drift_strength))
    n_train_B = n_train - n_train_A

    train_idx = np.concatenate([
        _stratified_sample_indices(y_all_int, idx_A, min(n_train_A, len(idx_A)), rng=rng),
        _stratified_sample_indices(y_all_int, idx_B, min(n_train_B, len(idx_B)), rng=rng),
    ])
    train_idx = np.unique(train_idx)

    remaining = np.setdiff1d(idx_all, train_idx, assume_unique=False)

    idx_A_rem = remaining[np.isin(clusters[remaining], list(domainA))]
    idx_B_rem = remaining[np.isin(clusters[remaining], list(domainB))]

    n_future = n_total - len(train_idx)
    n_future_B = int(round(n_future * drift_strength))
    n_future_A = n_future - n_future_B

    future_idx = np.concatenate([
        _stratified_sample_indices(y_all_int, idx_B_rem, min(n_future_B, len(idx_B_rem)), rng=rng),
        _stratified_sample_indices(y_all_int, idx_A_rem, min(n_future_A, len(idx_A_rem)), rng=rng),
    ])
    future_idx = np.unique(future_idx)

    if len(future_idx) < n_future:
        leftover = np.setdiff1d(remaining, future_idx, assume_unique=False)
        if len(leftover) > 0:
            extra = rng.choice(leftover, min(n_future - len(future_idx), len(leftover)), replace=False)
            future_idx = np.unique(np.concatenate([future_idx, extra]))

    df_train = df_clean.iloc[train_idx].copy()
    df_future = df_clean.iloc[future_idx].copy()

    df_val, df_test = train_test_split(
        df_future,
        test_size=(n_test / max(n_val + n_test, 1)),
        random_state=random_state,
        shuffle=True
    )

    return df_train, df_val, df_test


def split_domain_shift_quantile(
    df: pd.DataFrame,
    label_col: str,
    train_qmax: float,
    test_qmin: float,
    drift_features: Optional[List[str]],
    random_state: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    df_clean, X_all, y_all_str, feature_names = preprocess_aligned(df, label_col=label_col)
    y_all_int = LabelEncoder().fit_transform(y_all_str.to_numpy())

    if drift_features is None or len(drift_features) == 0:
        var = np.nanvar(X_all, axis=0)
        j = int(np.nanargmax(var))
        drift_features = [feature_names[j]]

    mask_A = np.ones(len(df_clean), dtype=bool)
    mask_B = np.ones(len(df_clean), dtype=bool)

    for fname in drift_features:
        if fname not in df_clean.columns:
            continue
        col = pd.to_numeric(df_clean[fname], errors="coerce").to_numpy()
        qA = np.nanquantile(col, train_qmax)
        qB = np.nanquantile(col, test_qmin)
        mask_A &= (col <= qA)
        mask_B &= (col >= qB)

    idx_all = np.arange(len(df_clean))
    idx_A = idx_all[mask_A]
    idx_B = idx_all[mask_B]

    if len(idx_A) < 100:
        idx_A = idx_all
    if len(idx_B) < 100:
        idx_B = idx_all

    n_total = len(idx_all)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = max(1, int(round(n_total * test_frac)))
        n_val = max(1, n_total - n_train - n_test)

    train_idx = _stratified_sample_indices(y_all_int, idx_A, min(n_train, len(idx_A)), rng=rng)
    train_idx = np.unique(train_idx)

    remaining = np.setdiff1d(idx_all, train_idx, assume_unique=False)

    idx_B_rem = np.intersect1d(idx_B, remaining, assume_unique=False)
    future_need = n_total - len(train_idx)

    future_idx = _stratified_sample_indices(y_all_int, idx_B_rem, min(future_need, len(idx_B_rem)), rng=rng)
    future_idx = np.unique(future_idx)

    if len(future_idx) < future_need:
        leftover = np.setdiff1d(remaining, future_idx, assume_unique=False)
        if len(leftover) > 0:
            extra = rng.choice(leftover, min(future_need - len(future_idx), len(leftover)), replace=False)
            future_idx = np.unique(np.concatenate([future_idx, extra]))

    df_train = df_clean.iloc[train_idx].copy()
    df_future = df_clean.iloc[future_idx].copy()

    df_val, df_test = train_test_split(
        df_future,
        test_size=(n_test / max(n_val + n_test, 1)),
        random_state=random_state,
        shuffle=True
    )

    return df_train, df_val, df_test


# ----------------------------
# Class-imbalance Domain Shift Splitter (ALFA testing)
# ----------------------------

def _define_minority_classes(
    y: pd.Series,
    minority_def: str = "pct_total",
    minority_percentile: float = 0.25,
    minority_threshold: int = 200,
    topk_majority: int = 2,
    minority_pct_threshold: float = 0.05
) -> Tuple[Dict[str, int], set, set]:
    counts = y.astype(str).value_counts()
    classes = counts.index.tolist()

    total = int(counts.sum()) if len(counts) else 0

    if len(classes) < 2:
        return counts.to_dict(), set(classes), set()

    if minority_def == "pct_total":
        if total <= 0:
            minority = set()
            majority = set(classes)
        else:
            minority = set(counts[(counts / total) < minority_pct_threshold].index.tolist())
            majority = set(classes) - minority

    elif minority_def == "percentile":
        n = len(classes)
        k = max(1, int(np.ceil(n * minority_percentile)))
        minority = set(counts.sort_values(ascending=True).head(k).index.tolist())
        majority = set(classes) - minority

    elif minority_def == "threshold":
        minority = set(counts[counts < minority_threshold].index.tolist())
        majority = set(classes) - minority

    elif minority_def == "topk_majority":
        majority = set(counts.head(topk_majority).index.tolist())
        minority = set(classes) - majority

    else:
        raise ValueError(f"Unknown minority_def: {minority_def}")

    if len(majority) == 0 and len(minority) > 0:
        one = next(iter(minority))
        minority.remove(one)
        majority.add(one)
    if len(minority) == 0 and len(majority) > 1:
        smallest = counts.sort_values(ascending=True).index[0]
        minority = {smallest}
        majority = set(classes) - minority

    return counts.to_dict(), majority, minority


def split_domain_shift_class_imbalance(
    df: pd.DataFrame,
    label_col: str,
    labeled_minority_frac: float,
    domainb_minority_frac: float,
    minority_def: str,
    minority_percentile: float,
    minority_threshold: int,
    topk_majority: int,
    random_state: int,
    class_shift_policy: str = "strict",
    allow_replacement: bool = False,
    train_size: int | None = None,
    train_frac: float = 0.7,
    val_frac_of_remaining: float = 0.5,
    minority_pct_threshold: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    rng = np.random.default_rng(random_state)

    df_clean, _, y_all_str, _ = preprocess_aligned(df, label_col=label_col)
    y_all_str = y_all_str.astype(str)

    counts_dict, majority_set, minority_set = _define_minority_classes(
        y_all_str,
        minority_def=minority_def,
        minority_percentile=minority_percentile,
        minority_threshold=minority_threshold,
        topk_majority=topk_majority,
        minority_pct_threshold=minority_pct_threshold,
    )

    idx_all = np.arange(len(df_clean))
    y_arr = y_all_str.to_numpy()

    maj_mask = np.isin(y_arr, list(majority_set))
    min_mask = np.isin(y_arr, list(minority_set))

    idx_maj = idx_all[maj_mask]
    idx_min = idx_all[min_mask]

    n_total = len(idx_all)

    if train_size is not None:
        n_train = int(train_size)
    else:
        n_train = int(round(n_total * float(train_frac)))

    if n_train < 1:
        n_train = 1
    if (not allow_replacement) and n_train >= n_total:
        n_train = max(1, n_total - 2)

    remaining_total = n_total - n_train
    if remaining_total < 2 and not allow_replacement:
        raise ValueError(
            f"Not enough samples remaining after train split to form val/test. "
            f"n_total={n_total}, n_train={n_train}. Reduce train_frac/train_size or enable --allow_replacement."
        )

    val_frac_of_remaining = float(np.clip(val_frac_of_remaining, 0.0, 1.0))
    n_future = remaining_total
    n_val = max(1, int(round(n_future * val_frac_of_remaining)))
    n_val = min(n_val, max(1, n_future - 1))
    n_test = max(1, n_future - n_val)

    classes_all = np.unique(y_arr)
    n_classes = len(classes_all)

    if not allow_replacement:
        counts_per_class = {c: int(np.sum(y_arr == c)) for c in classes_all}
        too_small = {c: n for c, n in counts_per_class.items() if n < 3}
        if len(too_small) > 0:
            raise RuntimeError(
                "Cannot enforce >=1 sample of each class in train/val/test WITHOUT replacement. "
                f"These classes have < 3 total samples: {too_small}. "
                "Enable --allow_replacement if you want to force the constraint anyway."
            )

    if (not allow_replacement) and (n_train < n_classes or n_val < n_classes or n_test < n_classes):
        raise RuntimeError(
            "Cannot enforce >=1 per class in each split WITHOUT replacement because one of the splits "
            f"is smaller than the number of classes. sizes(train/val/test)=({n_train},{n_val},{n_test}), "
            f"n_classes={n_classes}. Increase split sizes or enable --allow_replacement."
        )

    available = idx_all.copy()

    def _take_one_from_class(cls_label: str, pool_idx: np.ndarray) -> int | None:
        cls_idx = pool_idx[y_arr[pool_idx] == cls_label]
        if len(cls_idx) == 0:
            return None
        return int(rng.choice(cls_idx, size=1, replace=False)[0])

    seed_train, seed_val, seed_test = [], [], []

    for cls in classes_all:
        i = _take_one_from_class(cls, available)
        if i is None:
            if allow_replacement:
                full_cls_idx = idx_all[y_arr == cls]
                i = int(rng.choice(full_cls_idx, size=1, replace=True)[0])
            else:
                raise RuntimeError(f"Class {cls} missing during train seeding.")
        seed_train.append(i)
        if not allow_replacement:
            available = np.setdiff1d(available, np.array([i]), assume_unique=False)

        i = _take_one_from_class(cls, available)
        if i is None:
            if allow_replacement:
                full_cls_idx = idx_all[y_arr == cls]
                i = int(rng.choice(full_cls_idx, size=1, replace=True)[0])
            else:
                raise RuntimeError(f"Class {cls} missing during val seeding.")
        seed_val.append(i)
        if not allow_replacement:
            available = np.setdiff1d(available, np.array([i]), assume_unique=False)

        i = _take_one_from_class(cls, available)
        if i is None:
            if allow_replacement:
                full_cls_idx = idx_all[y_arr == cls]
                i = int(rng.choice(full_cls_idx, size=1, replace=True)[0])
            else:
                raise RuntimeError(f"Class {cls} missing during test seeding.")
        seed_test.append(i)
        if not allow_replacement:
            available = np.setdiff1d(available, np.array([i]), assume_unique=False)

    seed_train = np.array(seed_train, dtype=int)
    seed_val = np.array(seed_val, dtype=int)
    seed_test = np.array(seed_test, dtype=int)

    seeded_all = np.unique(np.concatenate([seed_train, seed_val, seed_test]))

    def _take(pool: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=int)
        if len(pool) == 0:
            return np.array([], dtype=int)
        if (not allow_replacement) and n > len(pool):
            n = len(pool)
        return rng.choice(pool, size=n, replace=allow_replacement)

    if not allow_replacement:
        maj_pool = np.setdiff1d(idx_maj, seeded_all, assume_unique=False)
        min_pool = np.setdiff1d(idx_min, seeded_all, assume_unique=False)
    else:
        maj_pool = idx_maj.copy()
        min_pool = idx_min.copy()

    def _count_minority(indices: np.ndarray) -> int:
        if len(indices) == 0:
            return 0
        return int(np.sum(np.isin(y_arr[indices], list(minority_set))))

    seed_train_min = _count_minority(seed_train)
    seed_val_min = _count_minority(seed_val)
    seed_test_min = _count_minority(seed_test)
    seed_future_min = seed_val_min + seed_test_min

    desired_train_min = int(round(n_train * float(labeled_minority_frac)))
    desired_train_maj = n_train - desired_train_min

    desired_future_min = int(round(n_future * float(domainb_minority_frac)))
    desired_future_maj = n_future - desired_future_min

    seed_train_maj = len(seed_train) - seed_train_min
    seed_future_maj = (len(seed_val) + len(seed_test)) - seed_future_min

    train_min_n = max(0, desired_train_min - seed_train_min)
    train_maj_n = max(0, desired_train_maj - seed_train_maj)

    future_min_n = max(0, desired_future_min - seed_future_min)
    future_maj_n = max(0, desired_future_maj - seed_future_maj)

    train_min = _take(min_pool, train_min_n)
    train_maj = _take(maj_pool, train_maj_n)

    train_idx = np.concatenate([seed_train, train_min, train_maj])

    if not allow_replacement:
        used = set(train_idx.tolist())
        maj_pool = np.array([i for i in maj_pool if int(i) not in used], dtype=int)
        min_pool = np.array([i for i in min_pool if int(i) not in used], dtype=int)

    if class_shift_policy == "strict" and len(np.unique(train_idx)) != n_train and not allow_replacement:
        raise RuntimeError(
            f"STRICT class_imbalance: could not build train of size {n_train} without overlap. "
            f"Got {len(np.unique(train_idx))}. "
            f"Try --class_shift_policy soft or --allow_replacement, or adjust fractions/sizes."
        )

    if class_shift_policy == "soft":
        uniq_train = np.unique(train_idx)
        if len(uniq_train) < n_train:
            remaining = np.setdiff1d(idx_all, uniq_train, assume_unique=False)
            need = n_train - len(uniq_train)
            if len(remaining) > 0:
                filler = rng.choice(remaining, size=min(need, len(remaining)), replace=allow_replacement)
                train_idx = np.concatenate([uniq_train, filler])
            else:
                train_idx = uniq_train
        else:
            train_idx = uniq_train
    else:
        train_idx = np.unique(train_idx) if not allow_replacement else train_idx

    future_min = _take(min_pool, future_min_n)
    future_maj = _take(maj_pool, future_maj_n)

    future_idx = np.concatenate([seed_val, seed_test, future_min, future_maj])

    if not allow_replacement:
        future_idx = np.setdiff1d(np.unique(future_idx), np.unique(train_idx), assume_unique=False)

    if class_shift_policy == "strict" and (not allow_replacement) and len(future_idx) != n_future:
        raise RuntimeError(
            f"STRICT class_imbalance: could not build future (val+test) of size {n_future}. Got {len(future_idx)}. "
            f"Try --class_shift_policy soft or --allow_replacement, or adjust fractions/sizes."
        )

    if class_shift_policy == "soft":
        uniq_train = np.unique(train_idx) if not allow_replacement else np.array(train_idx, dtype=int)
        uniq_future = np.unique(future_idx) if not allow_replacement else np.array(future_idx, dtype=int)

        if not allow_replacement and len(uniq_future) < n_future:
            remaining = np.setdiff1d(idx_all, np.unique(np.concatenate([uniq_train, uniq_future])), assume_unique=False)
            need = n_future - len(uniq_future)
            if len(remaining) > 0:
                filler = rng.choice(remaining, size=min(need, len(remaining)), replace=allow_replacement)
                uniq_future = np.unique(np.concatenate([uniq_future, filler]))
        future_idx = uniq_future
    else:
        future_idx = np.unique(future_idx) if not allow_replacement else future_idx

    if not allow_replacement:
        seed_val_u = np.unique(seed_val)
        seed_test_u = np.unique(seed_test)
        remaining_future = np.setdiff1d(future_idx, np.unique(np.concatenate([seed_val_u, seed_test_u])), assume_unique=False)

        need_val = max(0, n_val - len(seed_val_u))
        need_test = max(0, n_test - len(seed_test_u))

        if need_val + need_test > len(remaining_future):
            if class_shift_policy == "strict":
                raise RuntimeError(
                    f"Not enough future samples to fill val/test after seeding. "
                    f"need_val={need_val}, need_test={need_test}, remaining_future={len(remaining_future)}."
                )

        val_fill = rng.choice(remaining_future, size=min(need_val, len(remaining_future)), replace=False) if need_val > 0 else np.array([], dtype=int)
        remaining_after_val = np.setdiff1d(remaining_future, val_fill, assume_unique=False)

        test_fill = rng.choice(remaining_after_val, size=min(need_test, len(remaining_after_val)), replace=False) if need_test > 0 else np.array([], dtype=int)

        val_idx = np.unique(np.concatenate([seed_val_u, val_fill]))
        test_idx = np.unique(np.concatenate([seed_test_u, test_fill]))

        if class_shift_policy == "soft":
            if len(val_idx) < n_val:
                leftover = np.setdiff1d(remaining_after_val, test_fill, assume_unique=False)
                need = n_val - len(val_idx)
                if len(leftover) > 0:
                    extra = rng.choice(leftover, size=min(need, len(leftover)), replace=False)
                    val_idx = np.unique(np.concatenate([val_idx, extra]))
            if len(test_idx) < n_test:
                leftover = np.setdiff1d(remaining_after_val, test_fill, assume_unique=False)
                leftover = np.setdiff1d(leftover, val_idx, assume_unique=False)
                need = n_test - len(test_idx)
                if len(leftover) > 0:
                    extra = rng.choice(leftover, size=min(need, len(leftover)), replace=False)
                    test_idx = np.unique(np.concatenate([test_idx, extra]))
    else:
        future_candidates = np.array(future_idx, dtype=int)
        if len(future_candidates) == 0:
            future_candidates = idx_all

        need_val = max(0, n_val - len(seed_val))
        need_test = max(0, n_test - len(seed_test))

        val_fill = rng.choice(future_candidates, size=need_val, replace=True) if need_val > 0 else np.array([], dtype=int)
        test_fill = rng.choice(future_candidates, size=need_test, replace=True) if need_test > 0 else np.array([], dtype=int)

        val_idx = np.concatenate([seed_val, val_fill])
        test_idx = np.concatenate([seed_test, test_fill])

    df_train = df_clean.iloc[np.unique(train_idx) if not allow_replacement else train_idx].copy()
    df_val = df_clean.iloc[val_idx].copy()
    df_test = df_clean.iloc[test_idx].copy()

    shift_meta = {
        "shift_mode": "class_imbalance",
        "minority_def": minority_def,
        "minority_percentile": float(minority_percentile),
        "minority_threshold": int(minority_threshold),
        "topk_majority": int(topk_majority),
        "minority_pct_threshold": float(minority_pct_threshold),
        "labeled_minority_frac": float(labeled_minority_frac),
        "domainb_minority_frac": float(domainb_minority_frac),
        "class_shift_policy": class_shift_policy,
        "allow_replacement": bool(allow_replacement),
        "enforce_min_1_per_class_per_split": True,
        "majority_classes": sorted(list(majority_set)),
        "minority_classes": sorted(list(minority_set)),
        "class_counts_full": counts_dict,
        "sizes": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))}
    }

    return df_train, df_val, df_test, shift_meta


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Split data and save NPZ with optional discrete transform + CIC-like drift.'
    )

    parser.add_argument('--uci_num', default=None, type=int)

    parser.add_argument('--input_csv', required=False,
                        help='Path to the input CSV file (must contain a label column).')
    parser.add_argument("--unlabeled_csv", required=False,
                        help="Optional separate csv for unlabeled (keeps your existing behavior).",
                        default=False)

    parser.add_argument('--output_dir', required=True,
                        help='Directory where the NPZ files and label2id.json will be saved.')

    parser.add_argument('--label-col', default='Label',
                        help="Name of the label column (default: 'Label').")

    parser.add_argument('--random-state', type=int, default=42)

    parser.add_argument('--discrete_to_label', action='store_true',
                        help='Transform discrete columns to numeric using label encoding.')

    # NEW: drop classes (supports comma-separated names)
    parser.add_argument(
        '--drop_classes',
        type=str,
        default="",
        help='Comma-separated class labels to DROP from the dataset before splitting. '
             'Example: --drop_classes "2,3,DoS".'
    )

    # NEW: shift settings
    parser.add_argument('--shift_mode', type=str, default="none",
                        choices=["none", "cluster", "quantile", "class_imbalance"],
                        help="Artificial drift mode. Default: none")

    parser.add_argument('--train_frac', type=float, default=0.25, help="Fraction of rows used for train.")
    parser.add_argument('--val_frac', type=float, default=0.55, help="Fraction of rows used for validation.")
    parser.add_argument('--test_frac', type=float, default=0.20, help="Fraction of rows used for test.")

    parser.add_argument('--drift_strength', type=float, default=0.9,
                        help="Cluster shift purity (0..1). Higher = stronger domain split.")
    parser.add_argument('--n_clusters', type=int, default=12,
                        help="KMeans clusters for cluster shift.")
    parser.add_argument('--pca_dim', type=int, default=10,
                        help="PCA dimension for clustering (<=0 disables PCA).")

    parser.add_argument('--train_qmax', type=float, default=0.6,
                        help="Domain A definition uses <= train_qmax quantile.")
    parser.add_argument('--test_qmin', type=float, default=0.6,
                        help="Domain B definition uses >= test_qmin quantile.")
    parser.add_argument('--drift_features', type=str, default="",
                        help="Comma-separated feature names to drive quantile shift.")

    parser.add_argument('--minority_def', type=str, default="pct_total",
                        choices=["pct_total", "percentile", "threshold", "topk_majority"],
                        help="How to define minority classes for class_imbalance shift.")
    parser.add_argument('--minority_percentile', type=float, default=0.25,
                        help="If minority_def=percentile: bottom p of classes by count are minority.")
    parser.add_argument('--minority_threshold', type=int, default=200,
                        help="If minority_def=threshold: classes with count < threshold are minority.")
    parser.add_argument('--topk_majority', type=int, default=2,
                        help="If minority_def=topk_majority: top-k most frequent classes are majority; rest minority.")
    parser.add_argument('--minority_pct_threshold', type=float, default=0.20,
                        help="If minority_def=pct_total: classes with (count/total) < threshold are minority.")
    parser.add_argument('--train_size', type=int, default=None,
                        help="Optional absolute train size for class_imbalance. If set, overrides --train_frac.")
    parser.add_argument('--val_frac_of_remaining', type=float, default=0.65,
                        help="For class_imbalance: fraction of remaining (Domain B) assigned to validation; rest to test.")
    parser.add_argument('--labeled_minority_frac', type=float, default=0.05,
                        help="Target minority fraction in Domain A (train/labeled) for class_imbalance shift.")
    parser.add_argument('--domainb_minority_frac', type=float, default=0.80,
                        help="Target minority fraction in Domain B (val/test) for class_imbalance shift.")
    parser.add_argument('--class_shift_policy', type=str, default="soft",
                        choices=["strict", "soft"],
                        help="strict enforces requested fractions if possible; soft is best-effort.")
    parser.add_argument('--allow_replacement', action='store_true',
                        help="Allow sampling with replacement if a group runs out (class_imbalance shift).")

    args = parser.parse_args()

    out_dir = "data/" + str(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    discrete_cols = []
    if args.uci_num is None:
        cfg = CN()
        cfg.DATASET = get_dataset_config(args.output_dir)
        discrete_cols = cfg.DATASET.DISCRETE_FEATURES

    drop_classes = parse_drop_classes_arg(args.drop_classes)

    # ----------------------------
    # Load df
    # ----------------------------

    if args.uci_num is not None:
        dataset = fetch_ucirepo(id=args.uci_num)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        if args.label_col not in df.columns:
            df.rename(columns={df.columns[-1]: args.label_col}, inplace=True)
    else:
        df = pd.read_csv("raw_data/" + str(args.input_csv))

    # Optional discrete transform BEFORE splitting
    if args.discrete_to_label:
        if not discrete_cols:
            raise ValueError('No discrete features defined in cfg.DATASET.DISCRETE_FEATURES')
        df = transform_discrete(df, discrete_cols)

    # NEW: Drop classes from the primary df BEFORE splitting
    df = drop_classes_from_df(df, label_col=args.label_col, drop_classes=drop_classes, tag="primary_df")

    # ----------------------------
    # Splitting logic
    # ----------------------------

    if args.unlabeled_csv is not False:
        df_unlabeled = pd.read_csv("raw_data/" + str(args.unlabeled_csv))
        if args.discrete_to_label and discrete_cols:
            df_unlabeled = transform_discrete(df_unlabeled, discrete_cols)

        # NEW: Drop classes from unlabeled df too, so they never appear in final splits
        df_unlabeled = drop_classes_from_df(df_unlabeled, label_col=args.label_col, drop_classes=drop_classes, tag="unlabeled_df")

        df_train, _ = train_test_split(
            df, test_size=0.3, random_state=args.random_state, shuffle=True
        )
        df_val, df_test = train_test_split(
            df_unlabeled, test_size=0.3, random_state=args.random_state, shuffle=True
        )

        shift_settings = {
            "shift_mode": "external_unlabeled_csv",
            "note": "val/test come from separate CSV; no artificial shift applied.",
            "drop_classes": drop_classes
        }

    else:
        if args.shift_mode == "none":
            tf = float(args.train_frac)
            vf = float(args.val_frac)
            tef = float(args.test_frac)
            s = tf + vf + tef
            if s <= 0:
                tf, vf, tef = 0.7, 0.15, 0.15
                s = 1.0
            tf, vf, tef = tf / s, vf / s, tef / s

            df_train, df_remain = train_test_split(
                df, test_size=(1.0 - tf), random_state=args.random_state, shuffle=True
            )
            rem = max(1e-9, (vf + tef))
            test_size_rel = tef / rem
            df_val, df_test = train_test_split(
                df_remain, test_size=test_size_rel, random_state=args.random_state, shuffle=True
            )
            shift_settings = {"shift_mode": "none", "drop_classes": drop_classes}

        elif args.shift_mode == "cluster":
            pca_dim = None if args.pca_dim is None or args.pca_dim <= 0 else args.pca_dim
            df_train, df_val, df_test = split_domain_shift_cluster(
                df=df,
                label_col=args.label_col,
                drift_strength=args.drift_strength,
                n_clusters=args.n_clusters,
                pca_dim=pca_dim,
                random_state=args.random_state,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac
            )
            shift_settings = {
                "shift_mode": "cluster",
                "drift_strength": args.drift_strength,
                "n_clusters": args.n_clusters,
                "pca_dim": pca_dim,
                "drop_classes": drop_classes
            }

        elif args.shift_mode == "quantile":
            drift_feats = [s.strip() for s in args.drift_features.split(",") if s.strip()] if args.drift_features else None
            df_train, df_val, df_test = split_domain_shift_quantile(
                df=df,
                label_col=args.label_col,
                train_qmax=args.train_qmax,
                test_qmin=args.test_qmin,
                drift_features=drift_feats,
                random_state=args.random_state,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=args.test_frac
            )
            shift_settings = {
                "shift_mode": "quantile",
                "train_qmax": args.train_qmax,
                "test_qmin": args.test_qmin,
                "drift_features": args.drift_features,
                "drop_classes": drop_classes
            }

        elif args.shift_mode == "class_imbalance":
            df_train, df_val, df_test, shift_meta = split_domain_shift_class_imbalance(
                df=df,
                label_col=args.label_col,
                labeled_minority_frac=args.labeled_minority_frac,
                domainb_minority_frac=args.domainb_minority_frac,
                minority_def=args.minority_def,
                minority_percentile=args.minority_percentile,
                minority_threshold=args.minority_threshold,
                topk_majority=args.topk_majority,
                random_state=args.random_state,
                class_shift_policy=args.class_shift_policy,
                allow_replacement=args.allow_replacement,
                train_size=args.train_size,
                train_frac=args.train_frac,
                val_frac_of_remaining=args.val_frac_of_remaining,
                minority_pct_threshold=args.minority_pct_threshold
            )
            shift_meta["drop_classes"] = drop_classes
            shift_settings = shift_meta

        else:
            raise ValueError(f"Unknown shift_mode: {args.shift_mode}")

    # ----------------------------
    # Build label mapping
    # ----------------------------

    _, X_train, y_train_series, feat_names = preprocess_aligned(df_train, label_col=args.label_col)
    _, X_val, y_val_series, _ = preprocess_aligned(df_val, label_col=args.label_col)
    _, X_test, y_test_series, _ = preprocess_aligned(df_test, label_col=args.label_col)

    all_labels = pd.concat([y_train_series, y_val_series, y_test_series], ignore_index=True)
    class_names = sorted(all_labels.unique())
    if len(class_names) == 0:
        raise RuntimeError("No classes found after splitting (this should not happen).")
    label2id = {label: idx for idx, label in enumerate(class_names)}

    mapping_path = os.path.join(out_dir, 'label2id.json')
    with open(mapping_path, 'w') as f:
        json.dump(label2id, f, indent=2)

    y_train = y_train_series.map(label2id).to_numpy(dtype=np.int64)
    y_val   = y_val_series.map(label2id).to_numpy(dtype=np.int64)
    y_test  = y_test_series.map(label2id).to_numpy(dtype=np.int64)

    ts_train = np.arange(len(y_train), dtype=np.int64)
    ts_val   = np.arange(len(y_val), dtype=np.int64)
    ts_test  = np.arange(len(y_test), dtype=np.int64)

    np.savez(os.path.join(out_dir, 'train.npz'), feature=X_train, label=y_train, timestamp=ts_train)
    np.savez(os.path.join(out_dir, 'val.npz'),   feature=X_val,   label=y_val,   timestamp=ts_val)
    np.savez(os.path.join(out_dir, 'test.npz'),  feature=X_test,  label=y_test,  timestamp=ts_test)

    drift_report, drift_feat_df = compute_drift_report(
        X_train=X_train,
        X_test=X_test,
        feature_names=feat_names if len(feat_names) == X_train.shape[1] else [f"f{j}" for j in range(X_train.shape[1])]
    )

    drift_report_path = os.path.join(out_dir, "drift_report.json")
    with open(drift_report_path, "w") as f:
        json.dump({
            "split_definition": "CIC-like: train from Domain A (labeled), val/test from Domain B (future)",
            "settings": shift_settings,
            "drift_report_train_vs_test": drift_report,
            "sizes": {
                "train": int(X_train.shape[0]),
                "val": int(X_val.shape[0]),
                "test": int(X_test.shape[0])
            },
            "classes": {
                "n_classes": int(len(class_names)),
                "class_names": class_names
            }
        }, f, indent=2)

    drift_feat_path = os.path.join(out_dir, "drift_features.csv")
    drift_feat_df.to_csv(drift_feat_path, index=False)

    # ----------------------------------
    # Class distribution report (splits)
    # ----------------------------------
    def _safe_int(v):
        try:
            return int(v)
        except Exception:
            return None

    def _split_class_distribution(y_arr: np.ndarray, id2label: Dict[int, str]):
        y_arr = np.asarray(y_arr)
        vals, counts = np.unique(y_arr, return_counts=True)
        total = int(counts.sum())
        rows = []
        for v, c in sorted(zip(vals.tolist(), counts.tolist()), key=lambda t: (t[0] if isinstance(t[0], (int, np.integer)) else str(t[0]))):
            v_int = _safe_int(v)
            label = id2label.get(v_int, str(v)) if v_int is not None else str(v)
            rows.append({
                'class_id': v_int if v_int is not None else str(v),
                'class_label': label,
                'count': int(c),
                'fraction': (float(c) / total) if total > 0 else 0.0
            })
        return {'total': total, 'by_class': rows}

    id2label = {int(v): str(k) for k, v in label2id.items()}
    dist_train = _split_class_distribution(y_train, id2label)
    dist_val   = _split_class_distribution(y_val, id2label)
    dist_test  = _split_class_distribution(y_test, id2label)

    dist_report = {
        'train': dist_train,
        'val': dist_val,
        'test': dist_test,
    }

    dist_json_path = os.path.join(out_dir, 'split_class_distribution.json')
    with open(dist_json_path, 'w', encoding='utf-8') as f:
        json.dump(dist_report, f, indent=2)

    dist_rows = []
    for split_name, d in dist_report.items():
        for r in d['by_class']:
            dist_rows.append({
                'split': split_name,
                **r
            })
    dist_csv_path = os.path.join(out_dir, 'split_class_distribution.csv')
    pd.DataFrame(dist_rows).to_csv(dist_csv_path, index=False)

    print(f"\nSaved splits in '{out_dir}':")
    print(f"  train.npz: {X_train.shape[0]} samples, classes={len(class_names)}")
    print(f"  val.npz:   {X_val.shape[0]} samples (used as unlabeled pool in AL)")
    print(f"  test.npz:  {X_test.shape[0]} samples")
    print(f"  label2id:  {mapping_path}")
    print(f"  drift_report: {drift_report_path}")
    print(f"  drift_features: {drift_feat_path}")
    print(f"  class_distribution_json: {dist_json_path}")
    print(f"  class_distribution_csv:  {dist_csv_path}")
    if drop_classes:
        print(f"  dropped_classes: {drop_classes}")

    print("\nClass distribution (top 5 classes by fraction):")
    for _split in ["train", "val", "test"]:
        _rows = dist_report[_split]["by_class"]
        _top = sorted(_rows, key=lambda r: r["fraction"], reverse=True)[:5]
        _top_str = ", ".join([f"{r['class_label']}:{r['fraction']:.2f}" for r in _top])
        print(f"  {_split}: {_top_str}")

    print("\nDrift metrics (train vs test):")
    print(f"  PSI(mean/median/max)=({drift_report['psi_mean']:.4f}, {drift_report['psi_median']:.4f}, {drift_report['psi_max']:.4f})")
    print(f"  JS(mean/median/max)=({drift_report['js_mean']:.4f}, {drift_report['js_median']:.4f}, {drift_report['js_max']:.4f})")
    print(f"  MMD_RBF={drift_report.get('mmd_rbf', 0.0):.4f}")


if __name__ == '__main__':
    main()
