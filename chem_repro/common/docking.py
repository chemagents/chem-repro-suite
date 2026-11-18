"""Aggregate pose-recovery observations without implying pose generation or affinity validation."""

import numpy as np
import pandas as pd


def pose_recovery_summary(records: pd.DataFrame, *, rmsd_threshold: float, max_rank: int = 5):
    """Summarize a table whose RMSD is the minimum across the declared top-k poses.

    The first recovery rank was defined at rmsd_threshold by the source. It is
    insufficient to recompute rank-specific recovery at a different threshold.
    Exclusions stay in the panel table and never enter the evaluated denominator.
    """
    required = {"target", "evaluated", "best_top5_rmsd_A", "first_recovery_rank", "exclusion_reason"}
    if not required.issubset(records):
        raise ValueError(f"Missing pose-recovery columns: {sorted(required - set(records))}")
    if max_rank != 5 or not np.isfinite(rmsd_threshold) or rmsd_threshold <= 0:
        raise ValueError("This table stores best-top5 RMSD; require max_rank=5 and a positive threshold")
    if records.target.isna().any() or records.target.duplicated().any():
        raise ValueError("Require one nonmissing record per target")
    if not records.evaluated.isin([True, False]).all() or records.evaluated.isna().any():
        raise ValueError("evaluated must be an explicit Boolean")
    valid = records.loc[records.evaluated].copy()
    excluded = records.loc[~records.evaluated]
    if valid.empty or valid.best_top5_rmsd_A.isna().any():
        raise ValueError("Evaluated targets require RMSD observations")
    rmsd = valid.best_top5_rmsd_A.to_numpy(float)
    if not np.isfinite(rmsd).all() or (rmsd < 0).any():
        raise ValueError("RMSD must be finite and nonnegative")
    ranks = valid.first_recovery_rank
    given = ranks.dropna()
    if not ((given >= 1) & (given <= max_rank) & (given == np.floor(given))).all():
        raise ValueError("Recovery rank must be an integer in [1,5] or missing for failure")
    if not np.array_equal(ranks.notna().to_numpy(), rmsd <= rmsd_threshold):
        raise ValueError("Recovery ranks conflict with thresholded best-top5 RMSD")
    if excluded.exclusion_reason.fillna("").str.strip().eq("").any():
        raise ValueError("Every excluded target needs a reason")
    summary = pd.DataFrame(
        [
            {
                "top_k": k,
                "n_recovered": int((ranks <= k).sum()),
                "n_evaluated": len(valid),
                "fraction_recovered": float((ranks <= k).mean()),
            }
            for k in range(1, max_rank + 1)
        ]
    )
    return summary, valid.loc[ranks.isna()], excluded, float(np.median(rmsd))
