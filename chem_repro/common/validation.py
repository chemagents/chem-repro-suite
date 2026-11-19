"""Auditable regression validation shared by study-specific model adapters."""

import hashlib
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


def regression_model(name, seed):
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, max_depth=20, random_state=seed, n_jobs=2)
    if name == "ElasticNet":
        return ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=seed)
    raise ValueError("Unknown fixed model")


def regression_oof(x, y, groups, ids, *, model, bootstrap=1000, seed=42, fold_assignments=None):
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if not 100 <= bootstrap <= 1000:
        raise ValueError("Use 100–1000 bootstrap iterations")
    if len(x) != len(y) or len(groups) != len(y) or len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("Mismatched rows or duplicate observation IDs")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(set(groups)) < 5:
        raise ValueError("Finite model inputs and five distinct groups required")
    results = {}
    for scheme in ["random", "scaffold"]:
        if fold_assignments is None:
            folds = (KFold(5, shuffle=True, random_state=seed).split(x) if scheme == "random"
                     else GroupKFold(5).split(x, y, groups))
        else:
            if set(fold_assignments) != {"random", "scaffold"}:
                raise ValueError("Both reference split schemes are required")
            assignment = np.asarray(fold_assignments[scheme])
            if (assignment.shape != (len(y),) or not np.isin(assignment, range(5)).all()
                    or set(assignment.tolist()) != set(range(5))):
                raise ValueError("Reference folds must cover every row with five nonempty folds")
            folds = [(np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold))
                     for fold in range(5)]
        predictions = np.full(len(y), np.nan)
        fold_ids = np.full(len(y), -1, int)
        audit = []
        for fold, (train, test) in enumerate(folds):
            overlap = len(set(groups[train]) & set(groups[test]))
            if scheme == "scaffold" and overlap:
                raise ValueError("Scaffold groups leak across held-out split")
            estimator = regression_model(model, seed)
            estimator.fit(x[train], y[train])
            predictions[test] = estimator.predict(x[test])
            fold_ids[test] = fold
            audit.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "generic_scaffold_overlap": overlap,
                }
            )
        if not np.isfinite(predictions).all() or (fold_ids < 0).any():
            raise RuntimeError("Incomplete OOF predictions")
        rng = np.random.default_rng(seed)
        b = []
        for _ in range(bootstrap):
            sample = rng.integers(0, len(y), len(y))
            if np.ptp(y[sample]) > 0:
                b.append(r2_score(y[sample], predictions[sample]))
        registry = [
            {"observation_id": str(i), "fold": int(f), "generic_scaffold_group": str(g),
             "observed": float(v), "predicted": float(p)}
            for i, f, g, v, p in zip(ids, fold_ids, groups, y, predictions)
        ]
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True, allow_nan=False).encode()).hexdigest()
        interval = np.quantile(b, [0.025, 0.975])
        results[scheme] = {
            "r2": float(r2_score(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2_ci_low": float(interval[0]),
            "r2_ci_high": float(interval[1]),
            "oof_sha256": digest,
            "fold_audit": audit,
            "oof": registry,
        }
    return results
