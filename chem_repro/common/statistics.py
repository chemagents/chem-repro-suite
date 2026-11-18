"""Shared numerical pool; article adapters own cohort selection and protocol choices."""

from __future__ import annotations

import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def bh_fdr(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite values in [0,1]")
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(
        1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    )
    return q


class MeanPairwiseKS:
    """Tie-aware ECDF distances with a fixed sort, reusable for label permutations."""

    def __init__(self, values, groups):
        from itertools import combinations

        x = np.asarray(values, float)
        if not np.isfinite(x).all():
            raise ValueError("KS values must be finite")
        self.order = np.argsort(x, kind="mergesort")
        ordered = x[self.order]
        self.ends = np.r_[ordered[:-1] != ordered[1:], True]
        self.groups = list(groups)
        self.pairs = list(combinations(range(len(groups)), 2))
        if not self.pairs:
            raise ValueError("At least two groups required")

    def __call__(self, labels):
        labels = np.asarray(labels)[self.order]
        indicators = np.column_stack([labels == g for g in self.groups])
        sizes = indicators.sum(axis=0)
        if (sizes < 2).any():
            raise ValueError("Every group needs at least two observations")
        ecdf = (indicators.cumsum(axis=0) / sizes)[self.ends]
        return float(np.mean([np.abs(ecdf[:, a] - ecdf[:, b]).max() for a, b in self.pairs]))


def permutation_separability(axes: dict, labels, groups, *, permutations: int, seed: int):
    if not 99 <= permutations <= 10000:
        raise ValueError("Use 99–10000 permutations")
    rng = np.random.default_rng(seed)
    rows = []
    for axis, values in axes.items():
        ks = MeanPairwiseKS(values, groups)
        observed = ks(labels)
        null = np.array([ks(rng.permutation(labels)) for _ in range(permutations)])
        rows.append(
            {
                "axis": axis,
                "observed_ks": observed,
                "null_mean_ks": float(null.mean()),
                "null_low95": float(np.quantile(null, 0.025)),
                "null_high95": float(np.quantile(null, 0.975)),
                "p_permutation": (int((null >= observed).sum()) + 1) / (permutations + 1),
            }
        )
    for row, q in zip(rows, bh_fdr([r["p_permutation"] for r in rows])):
        row["p_fdr"] = float(q)
    return rows


def distribution(values) -> dict:
    x = np.asarray(values, dtype=float)
    valid = x[np.isfinite(x)]
    if not len(valid):
        return {
            "n": 0,
            "missing": len(x),
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "skewness": None,
        }
    return {
        "n": len(valid),
        "missing": len(x) - len(valid),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "q25": float(np.quantile(valid, 0.25)),
        "q75": float(np.quantile(valid, 0.75)),
        "skewness": float(skew(valid, bias=True)) if len(valid) > 2 and np.ptp(valid) > 0 else None,
    }


def profile_pca(matrix, *, standardize: bool = True, threshold: float = 0.8) -> dict:
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("PCA requires a finite 2-D matrix with at least two rows and columns")
    if not 0 < threshold <= 1 or not np.any(np.var(x, axis=0) > 0):
        raise ValueError("PCA needs variation and a threshold in (0,1]")
    z = StandardScaler().fit_transform(x) if standardize else x
    model = PCA(svd_solver="full").fit(z)
    variance = model.explained_variance_ratio_
    return {
        "scores": model.transform(z),
        "loadings": model.components_.T,
        "explained_variance_ratio": variance,
        "components_to_threshold": min(
            len(variance), int(np.searchsorted(np.cumsum(variance), threshold)) + 1
        ),
    }
