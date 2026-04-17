# pool_filters_cumulative.py
# Compact pool-filtering classes for Active Learning on tabular data.
#
# API (all methods):
#   subset(X_l, y_l, X_u) -> indices into X_u
#
# Includes:
#   - anchoral : Anchored KNN subpool (A anchors/class via k-means++ seeding)
#   - RandSub  : Random subpool (fixed size M)
#   - SEALS    : KNN-union with optional cumulative behavior across rounds
#
# Demo:
#   run_demo_round(X_l, y_l, X_u, y_u, M=1000)

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import kmeans_plusplus
from sklearn.neighbors import NearestNeighbors


# ------------------ utils ------------------
def _to_np(a):
    """Coerce arrays / pandas Series to a dense numpy array."""
    if hasattr(a, "to_numpy"):
        return a.to_numpy()
    return np.asarray(a)

def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    return Z / n

def print_distribution(y_values, idx, title: str):
    """
    Safe label distribution printer.
    - Converts y_values to numpy first (so idx acts as positional indexing)
    - Clips any out-of-range idx just in case
    """
    y_arr = _to_np(y_values)
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        print(f"{title}: (empty)")
        return
    idx = idx[(idx >= 0) and (idx < len(y_arr))]
    if idx.size == 0:
        print(f"{title}: (empty after index bounds check)")
        return
    labels, counts = np.unique(y_arr[idx], return_counts=True)
    dist = dict(zip(labels.tolist(), counts.tolist()))
    print(f"{title} (size={idx.size}): {dist}")


# ------------------ AnchorAL ------------------
class anchoral:
    """
    AnchorAL subpool:
      1) Pick A anchors/class from labeled set (k-means++ in rep space)
      2) Retrieve KNN of anchors from unlabeled set
      3) Average duplicate scores, return top-M indices into X_u
    """
    def __init__(self, A=10, K=50, M=2700, rep="scaled", pca_dim=64, metric="cosine", random_state=42):
        self.A, self.K, self.M = A, K, M
        self.rep, self.pca_dim, self.metric = rep, pca_dim, metric
        self.rng = np.random.RandomState(random_state)

    def _fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = _to_np(X)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        if self.rep == "pca":
            pca = PCA(n_components=min(self.pca_dim, Xs.shape[1]),
                      random_state=self.rng.randint(1, 1_000_000)).fit(Xs)
            Z = pca.transform(Xs)
        else:
            pca, Z = None, Xs
        if self.metric == "cosine":
            Z = _normalize_rows(Z)
        # We don't need to reuse scaler/pca across calls for AnchorAL; keep it simple
        return Z

    def _extract_anchors(self, Z_l: np.ndarray, y_l: np.ndarray) -> np.ndarray:
        anchors = []
        y_l = _to_np(y_l)
        for c in np.unique(y_l):
            mask = (y_l == c)
            Zc = Z_l[mask]
            idx_c = np.where(mask)[0]
            if Zc.shape[0] <= self.A:
                anchors.extend(idx_c.tolist())
            else:
                _, seeds = kmeans_plusplus(Zc, n_clusters=self.A, random_state=self.rng)
                anchors.extend(idx_c[seeds].tolist())
        return np.asarray(anchors, dtype=int)

    def _knn_in_unlabeled(self, z_query: np.ndarray, Z_u: np.ndarray, k: int):
        nn = NearestNeighbors(metric="cosine" if self.metric == "cosine" else "euclidean").fit(Z_u)
        dists, idx = nn.kneighbors(z_query[None, :], n_neighbors=min(k, len(Z_u)), return_distance=True)
        dists, idx = dists[0], idx[0]
        sims = 1.0 - dists if self.metric == "cosine" else -dists
        return idx, sims

    def subset(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> np.ndarray:
        Z_l = self._fit_transform(X_l)
        Z_u = self._fit_transform(X_u)

        anchor_pos = self._extract_anchors(Z_l, y_l)
        if anchor_pos.size == 0:
            # fallback: random labeled positions as anchors
            take = min(self.A, len(Z_l))
            anchor_pos = self.rng.choice(len(Z_l), size=take, replace=False)

        scores: Dict[int, list] = {}
        for a in anchor_pos:
            idxs, sims = self._knn_in_unlabeled(Z_l[a], Z_u, self.K)
            for i, s in zip(idxs, sims):
                scores.setdefault(int(i), []).append(float(s))

        if not scores:
            # fallback: random unlabeled positions
            return self.rng.choice(np.arange(len(Z_u)), size=min(self.M, len(Z_u)), replace=False)

        items = np.array([(i, np.mean(v)) for i, v in scores.items()], dtype=object)
        order = np.argsort([-x for x in items[:, 1].astype(float)])
        top = items[order][: min(self.M, items.shape[0])]
        return top[:, 0].astype(int)


# ------------------ RandSub ------------------
class RandSub:
    """Random subset of X_u of size up to M. Ignores labeled data."""
    def __init__(self, M=2700, random_state=987):
        self.M = M
        self.rng = np.random.RandomState(random_state)

    def subset(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> np.ndarray:
        X_u = _to_np(X_u)
        m = min(self.M, len(X_u))
        return self.rng.choice(np.arange(len(X_u)), size=m, replace=True)


# ------------------ SEALS (cumulative-capable) ------------------
class SEALS:
    """
    SEALS-like:
      - For each labeled sample, take its K nearest neighbors in the unlabeled pool.
      - Union them (dedup), cap to M.
      - If cumulative=True, union with previous rounds' neighbors (stateful).
    """
    def __init__(self, K=50, M=10000, rep="scaled", pca_dim=64, metric="cosine", cumulative=True):
        self.K, self.M, self.rep, self.pca_dim, self.metric = K, M, rep, pca_dim, metric
        self.cumulative = cumulative
        self._cum_ids: Optional[np.ndarray] = None

    def _fit_shared(self, X_l, X_u):
        X_l = _to_np(X_l)
        X_u = _to_np(X_u)
        X_all = np.vstack([X_l, X_u])
        scaler = StandardScaler().fit(X_all)
        Xs = scaler.transform(X_all)
        if self.rep == "pca":
            pca = PCA(n_components=min(self.pca_dim, Xs.shape[1])).fit(Xs)
        else:
            pca = None
        return scaler, pca

    def _transform(self, X, scaler, pca):
        X = _to_np(X)
        Xs = scaler.transform(X)
        Z = pca.transform(Xs) if pca is not None else Xs
        if self.metric == "cosine":
            Z = _normalize_rows(Z)
        return Z

    def subset(self, X_l: np.ndarray, y_l: np.ndarray, X_u: np.ndarray) -> np.ndarray:
        scaler, pca = self._fit_shared(X_l, X_u)
        Z_l = self._transform(X_l, scaler, pca)
        Z_u = self._transform(X_u, scaler, pca)

        nn = NearestNeighbors(metric="cosine" if self.metric == "cosine" else "euclidean").fit(Z_u)

        all_ids = []
        for i in range(Z_l.shape[0]):
            _, idx = nn.kneighbors(Z_l[i][None, :], n_neighbors=min(self.K, len(Z_u)), return_distance=True)
            all_ids.extend(idx[0].tolist())

        if not all_ids:
            return np.array([], dtype=int)

        ids, counts = np.unique(np.array(all_ids, dtype=int), return_counts=True)
        order = np.argsort(-counts)
        ids = ids[order]

        if self.cumulative and self._cum_ids is not None and self._cum_ids.size > 0:
            ids = np.unique(np.concatenate([self._cum_ids, ids], axis=0))

        if ids.size > self.M:
            ids = ids[: self.M]

        if self.cumulative:
            self._cum_ids = ids.copy()

        return ids


# ------------------ Demo (explicit labeled/unlabeled) ------------------
def run_demo_round(X_l, y_l, X_u, y_u, M=1000):
    """
    Takes explicit labeled and unlabeled sets.
      X_l, y_l: labeled data + labels
      X_u, y_u: unlabeled data + labels (labels used only for distribution printing)
    Returns dict of index arrays (relative to X_u).
    """
    anch = anchoral(A=10, K=50, M=M, rep="scaled", metric="cosine")
    rs   = RandSub(M=M, random_state=42)
    seals= SEALS(K=50, M=M, rep="scaled", metric="cosine", cumulative=True)

    idx_a = anch.subset(X_l, y_l, X_u)
    idx_r = rs.subset(X_l, y_l, X_u)
    idx_s = seals.subset(X_l, y_l, X_u)

    print_distribution(y_u, idx_a, "AnchorAL subset")
    print_distribution(y_u, idx_r, "RandSub subset")
    print_distribution(y_u, idx_s, "SEALS (cumulative) subset")

    return {"anchoral_idx_u": idx_a, "randsub_idx_u": idx_r, "seals_idx_u": idx_s}




ArrayLike = Union[np.ndarray]

def _as_np(x: ArrayLike) -> np.ndarray:
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)

def _print_distribution(y_values, idx, name: str):
    y_arr = _as_np(y_values)
    vals, cnts = np.unique(y_arr[idx], return_counts=True)
    dist = dict(zip(vals.tolist(), cnts.tolist()))
    print(f"{name} (n={len(idx)}): {dist}")

def subpool_anchoral(
    X_l: ArrayLike,
    y_l: ArrayLike,
    X_u: ArrayLike,
    y_u: ArrayLike,
    *,
    A: int = 10,
    K: int = 50,
    M: int = 2600,
    rep: str = "scaled",
    pca_dim: int = 64,
    metric: str = "cosine",
    random_state: int = 42,
    return_indices: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    AnchorAL subpool. Returns (X_u_subset, y_u_subset) by default,
    or (X_u_subset, y_u_subset, idx) if return_indices=True.
    """
    X_l, y_l, X_u = _as_np(X_l), _as_np(y_l), _as_np(X_u)
    y_u = _as_np(y_u)
    al = anchoral(A=A, K=K, M=M, rep=rep, pca_dim=pca_dim, metric=metric, random_state=random_state)
    idx = al.subset(X_l, y_l, X_u)
    _print_distribution(y_u, idx, "AnchorAL")
    X_sub, y_sub = X_u[idx], y_u[idx]
    return (X_sub, y_sub, idx) if return_indices else (X_sub, y_sub)

def subpool_randsub(
    X_l: ArrayLike,
    y_l: ArrayLike,
    X_u: ArrayLike,
    y_u: ArrayLike,
    *,
    M: int = 2600,
    random_state: int = 42,
    return_indices: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Random subpool of size up to M from X_u.
    Returns (X_u_subset, y_u_subset) or (..., idx) if return_indices=True.
    """
    X_u = _as_np(X_u)
    y_u = _as_np(y_u)
    rs = RandSub(M=M, random_state=random_state)
    idx = rs.subset(_as_np(X_l), _as_np(y_l), X_u)
    _print_distribution(y_u, idx, "RandSub")
    X_sub, y_sub = X_u[idx], y_u[idx]
    return (X_sub, y_sub, idx) if return_indices else (X_sub, y_sub)

def subpool_seals(
    X_l: ArrayLike,
    y_l: ArrayLike,
    X_u: ArrayLike,
    y_u: ArrayLike,
    *,
    K: int = 50,
    M: int = 2600,
    rep: str = "scaled",
    pca_dim: int = 64,
    metric: str = "cosine",
    cumulative: bool = False,
    return_indices: bool = False,
    _seals_obj: Optional[SEALS] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    SEALS subpool: union of KNN from each labeled sample, capped at M.
    Returns (X_u_subset, y_u_subset) or (..., idx) if return_indices=True.
    """
    X_l, y_l, X_u = _as_np(X_l), _as_np(y_l), _as_np(X_u)
    y_u = _as_np(y_u)
    seals = _seals_obj or SEALS(K=K, M=M, rep=rep, pca_dim=pca_dim, metric=metric, cumulative=cumulative)
    idx = seals.subset(X_l, y_l, X_u)
    _print_distribution(y_u, idx, "SEALS")
    X_sub, y_sub = X_u[idx], y_u[idx]
    return (X_sub, y_sub, idx) if return_indices else (X_sub, y_sub)