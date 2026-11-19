"""Summarize complete matched predictions without pooling repeated holdouts as OOF."""

import numpy as np
import pandas as pd

from .classification import binary_metrics


MEASURES = ("roc_auc", "average_precision", "pr_auc_trapezoidal", "fpr", "recall",
            "predicted_positive_fraction")


def summarize_matched_predictions(predictions, recorded_metrics, *, branches, candidate, source_ids):
    """Validate all recorded folds, then report mean/sample SD and two coverage denominators.

    Coverage of declared held-out instances and coverage of the labeled source population
    are distinct. Repeated holdouts can have complete scoring coverage without covering
    every source record. No confidence interval or independence claim is made.
    """
    if candidate not in branches or len(branches) < 2:
        raise ValueError("Declare a candidate and at least one reference branch")
    source_ids = set(source_ids)
    if not source_ids:
        raise ValueError("Source population is empty")
    records = pd.DataFrame(recorded_metrics)
    records = records.loc[records.scope == "fold"].copy()
    if records.empty or records.duplicated(["scheme", "fold", "branch"]).any():
        raise ValueError("Require unique complete fold/branch metrics")
    if set(records.scheme) != set(predictions):
        raise ValueError("Predictions and recorded schemes differ")
    checked, coverage, effects = [], [], []
    for scheme, data in predictions.items():
        frame = pd.DataFrame(data)
        if frame.empty or frame[["fold", "ligand_id"]].isna().any().any():
            raise ValueError("Missing prediction identities or folds")
        if frame.duplicated(["fold", "ligand_id"]).any():
            raise ValueError("Duplicate held-out identities within a fold")
        if not set(frame.ligand_id).issubset(source_ids):
            raise ValueError("Held-out identities outside the declared source")
        expected = records.loc[records.scheme == scheme]
        if set(frame.fold) != set(expected.fold):
            raise ValueError("Missing prediction folds")
        expected_instances = 0
        for fold, group in frame.groupby("fold", sort=True):
            if not isinstance(fold, (int, float, np.integer, np.floating)) or int(fold) != fold:
                raise ValueError("Fold indices must be integers")
            if set(group.activity) != {0, 1}:
                raise ValueError("Require exact binary labels without coercion")
            stored = expected.loc[expected.fold == fold].set_index("branch")
            if set(stored.index) != set(branches):
                raise ValueError("Missing matched branch metrics")
            counts = stored.n_test.unique()
            thresholds = stored.threshold.unique()
            if len(counts) != 1 or counts[0] != len(group) or len(thresholds) != 1:
                raise ValueError("Held-out counts or thresholds differ between branches")
            expected_instances += int(counts[0])
            values = {}
            for branch, column in branches.items():
                value = binary_metrics(group.activity.to_numpy(int), group[column].to_numpy(float),
                                       threshold=float(thresholds[0]))
                for name in (*MEASURES, "n_test", "n_positive", "n_negative", "tn", "fp", "fn", "tp"):
                    if not np.isclose(value[name], stored.loc[branch, name], rtol=0, atol=1e-12):
                        raise ValueError("Recorded metric differs from matched scores: " + name)
                checked.append({"scheme": scheme, "fold": int(fold), "branch": branch, **value})
                values[branch] = value
            for reference in branches:
                if reference == candidate:
                    continue
                effects.append({"scheme": scheme, "fold": int(fold), "candidate": candidate,
                                "reference": reference,
                                **{"delta_" + key: values[candidate][key] - values[reference][key]
                                   for key in MEASURES}})
        coverage.append({"scheme": scheme, "matched_scored_test_instances": len(frame),
                         "declared_test_instances": expected_instances,
                         "test_instance_coverage": len(frame) / expected_instances,
                         "unique_test_records": frame.ligand_id.nunique(),
                         "labeled_source_records": len(source_ids),
                         "source_record_coverage": frame.ligand_id.nunique() / len(source_ids),
                         "missing_branch_scores": int(frame[list(branches.values())].isna().sum().sum()),
                         "n_splits": frame.fold.nunique()})
    evaluated = pd.DataFrame(checked)
    summaries = []
    for (scheme, branch), group in evaluated.groupby(["scheme", "branch"], sort=True):
        summaries.append({"scheme": scheme, "branch": branch, "n_splits": len(group),
                          **{"mean_" + key: float(group[key].mean()) for key in MEASURES},
                          **{"sd_" + key: float(group[key].std(ddof=1)) if len(group) > 1 else None
                             for key in MEASURES},
                          **{"sum_" + key: int(group[key].sum()) for key in ("tn", "fp", "fn", "tp", "n_test")}})
    deltas = pd.DataFrame(effects)
    comparisons = []
    for (scheme, cand, reference), group in deltas.groupby(["scheme", "candidate", "reference"], sort=True):
        ref_fpr = evaluated.loc[(evaluated.scheme == scheme) & (evaluated.branch == reference), "fpr"].mean()
        delta_fpr = group.delta_fpr.mean()
        comparisons.append({"scheme": scheme, "candidate": cand, "reference": reference,
                            "n_paired_splits": len(group),
                            **{"mean_delta_" + key: float(group["delta_" + key].mean()) for key in MEASURES},
                            "relative_fpr_reduction_of_means": float(-delta_fpr / ref_fpr) if ref_fpr else None})
    return {"fold_summary": pd.DataFrame(summaries), "paired_summary": pd.DataFrame(comparisons),
            "scoring_coverage": pd.DataFrame(coverage), "paired_fold_differences": deltas}
