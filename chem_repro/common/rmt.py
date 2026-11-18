"""Train-only residue-energy selection, IJMS 2026 27 8000 Appendix A.

Numerical settings absent from the paper are explicit constructor parameters.
No outer test observations or labels enter fit. The diagnostic inner ROC-AUC
chooses m; it is not an unbiased estimate of predictive generalization.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _matrix(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not x.shape[1] or np.isinf(x).any():
        raise ValueError("Require a 2-D residue matrix without infinite energies")
    return np.where(np.isnan(x), 0.0, x)


@dataclass
class ResidueSelection:
    selected_indices: np.ndarray
    feature_names: tuple[str, ...]
    ranking: pd.DataFrame
    scan: pd.DataFrame
    spectrum: pd.DataFrame
    diagnostics: dict

    def transform(self, values, feature_names):
        if tuple(feature_names) != self.feature_names:
            raise ValueError("Residue columns must match the fitted names and order")
        x = _matrix(values)
        if x.shape[1] != len(self.feature_names):
            raise ValueError("Residue feature count mismatch")
        # Deliberately retain raw energies, not standardized or eigen-reconstructed features.
        return x[:, self.selected_indices]


def fit_residue_selector(
    x_train,
    y_train,
    feature_names,
    *,
    inner_splits=5,
    inner_test_size=0.2,
    seed=42,
    contribution_fraction=0.98,
    m_values=None,
    logistic_c=1.0,
):
    x = _matrix(x_train)
    y = np.asarray(y_train)
    n, p = x.shape
    names = tuple(feature_names)
    if n < 4 or y.shape != (n,) or set(y.tolist()) != {0, 1}:
        raise ValueError("Require aligned binary labels with both classes")
    if len(names) != p or len(set(names)) != p:
        raise ValueError("Require unique residue names matching matrix columns")
    if not 0 < contribution_fraction <= 1 or not 0 < inner_test_size < 1:
        raise ValueError("Invalid contribution fraction or inner test size")
    if not isinstance(inner_splits, int) or inner_splits < 2 or logistic_c <= 0:
        raise ValueError("Require at least two inner splits and positive LR C")
    candidates = list(range(1, p + 1)) if m_values is None else sorted(set(m_values))
    if not candidates or any(not isinstance(m, int) or not 1 <= m <= p for m in candidates):
        raise ValueError("Candidate m must be integers in [1, n_residues]")
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    constant = sd == 0
    sd[constant] = 1.0
    z = (x - mean) / sd
    eigenvalues, eigenvectors = np.linalg.eigh(z.T @ z / (n - 1))
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    threshold = float((1 + np.sqrt(p / n)) ** 2)
    signal = eigenvalues > threshold
    used = signal.copy()
    fallback = not signal.any()
    if fallback:
        used[0] = True
    scores = np.sum(eigenvectors[:, used] ** 2 * eigenvalues[used], axis=1)
    ranked = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(x[:, ranked], axis=1)
    folds = list(
        StratifiedShuffleSplit(n_splits=inner_splits, test_size=inner_test_size, random_state=seed).split(
            x, y
        )
    )
    scan = []
    for m in candidates:
        support = cumulative[:, m - 1 : m]
        aucs = []
        for train, valid in folds:
            lr = LogisticRegression(C=logistic_c, solver="lbfgs", max_iter=10000, random_state=seed)
            lr.fit(support[train], y[train])
            aucs.append(float(roc_auc_score(y[valid], lr.predict_proba(support[valid])[:, 1])))
        scan.append(
            {
                "m": m,
                "mean_inner_roc_auc": float(np.mean(aucs)),
                "std_inner_roc_auc": float(np.std(aucs, ddof=1)),
                "fold_roc_auc": aucs,
            }
        )
    scan = pd.DataFrame(scan)
    # Exact ties select the smaller m, declared in provenance rather than inferred from the paper.
    m_star = int(scan.iloc[int(np.argmax(scan.mean_inner_roc_auc))].m)
    top = ranked[:m_star]
    active_mean, inactive_mean = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    contribution = np.maximum(np.abs(active_mean), np.abs(inactive_mean))
    energy_order = top[np.argsort(-contribution[top], kind="stable")]
    total = float(contribution[top].sum())
    if total <= 0:
        raise ValueError("Selected residues have zero class-mean energetic contribution; 98% gate undefined")
    cumulative_fraction = np.cumsum(contribution[energy_order]) / total
    k = min(m_star, int(np.searchsorted(cumulative_fraction, contribution_fraction)) + 1)
    selected = energy_order[:k]
    rank = np.empty(p, int)
    rank[ranked] = np.arange(1, p + 1)
    within_rank = np.full(p, np.nan)
    within_rank[energy_order] = np.arange(1, m_star + 1)
    within_fraction = np.full(p, np.nan)
    within_fraction[energy_order] = cumulative_fraction
    ranking = (
        pd.DataFrame(
            {
                "residue": names,
                "rmt_score": scores,
                "rmt_rank": rank,
                "train_mean": mean,
                "train_sample_std": np.where(constant, 0, sd),
                "constant_training_feature": constant,
                "active_mean_energy": active_mean,
                "inactive_mean_energy": inactive_mean,
                "energetic_contribution": contribution,
                "in_top_m": np.isin(np.arange(p), top),
                "energy_rank_within_top_m": within_rank,
                "cumulative_contribution_fraction": within_fraction,
                "selected": np.isin(np.arange(p), selected),
            }
        )
        .sort_values("rmt_rank")
        .reset_index(drop=True)
    )
    return ResidueSelection(
        selected_indices=selected,
        feature_names=names,
        ranking=ranking,
        scan=scan,
        spectrum=pd.DataFrame(
            {
                "component": np.arange(1, p + 1),
                "eigenvalue": eigenvalues,
                "above_mp_edge": signal,
                "used_for_rmt_score": used,
            }
        ),
        diagnostics={
            "n_train": n,
            "n_residues": p,
            "q": p / n,
            "lambda_plus": threshold,
            "signal_components": int(signal.sum()),
            "leading_component_fallback": fallback,
            "m_star": m_star,
            "selected_residues": k,
            "retained_contribution_fraction": float(cumulative_fraction[k - 1]),
            "zero_imputed_training_cells": int(np.isnan(np.asarray(x_train, float)).sum()),
        },
    )
