"""Reported primary performance, separate from S4 and newly reconstructed validation."""

import pandas as pd

from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, provenance, table


def computed_primary_comparison(output, reference):
    """Compare only named estimators and like aggregations; never use a proxy HGB row.

    Numerical differences do not certify protocol equivalence or successful reproduction.
    The article's primary values are split means, so pooled OOF is kept out of this table.
    """
    candidates = []
    tabs = output.answer.tables
    if "fold_summary" in tabs:
        summary = tabs["fold_summary"]
        if summary.is_preview:
            raise ValueError("Primary comparison requires the complete split summary")
        for row in summary.rows:
            branch = row.get("branch")
            if branch in {"structure", "product", "rte"}:
                name = {"structure":"DMPNN", "product":"product", "rte":"RTE_ALL6"}[branch]
                candidates.append((name, row.get("scheme"), row, "sd_", row.get("n_splits")))
            elif row.get("target") == "ALL6" and row.get("model") == "CatBoost":
                candidates.append(("RTE_ALL6", output.metadata.protocol_parameters.get("protocol", "paper_kfold10"),
                                   row, "std_", row.get("n_folds")))
    if "validation_summary" in tabs:
        summary = tabs["validation_summary"]
        if summary.is_preview:
            raise ValueError("Primary comparison requires the complete structural summary")
        for row in summary.rows:
            converted = {"mean_roc_auc":row.get("roc_auc_fold_mean"), "sd_roc_auc":row.get("roc_auc_fold_std"),
                         "mean_fpr":row.get("fpr_fold_mean"), "sd_fpr":row.get("fpr_fold_std")}
            candidates.append(("DMPNN",row["scheme"],converted,"sd_",row["n_fits"]))
    comparisons=[]
    for name,scheme,values,sd_prefix,n in candidates:
        for ref in reference.loc[reference.model==name].itertuples():
            key={"ROC_AUC":"roc_auc", "FPR":"fpr"}[ref.metric]
            actual=values.get("mean_"+key)
            if actual is None:
                continue
            comparisons.append({"model":name,"scheme":scheme,"metric":ref.metric,"n_splits":n,
                "aggregation":"split_mean_and_sample_sd", "computed_mean":actual,
                "computed_sd":values.get(sd_prefix+key),"paper_mean":ref.comparison_mean,
                "paper_sd":ref.comparison_sd,"unit":"fraction",
                "computed_minus_paper":actual-ref.comparison_mean,
                "difference_percentage_points":100*(actual-ref.comparison_mean) if key=="fpr" else None,
                "exact_author_protocol_recovered":False})
    return comparisons


def attach_primary_validation(output, models):
    source = provenance("cannabis_candidates", "cannabis.primary_performance_transcription.v1")
    frame = pd.read_csv(DATA_ROOT / "cannabis_candidates/paper_primary_validation.csv")
    if not models or set(models) - set(frame.model):
        raise ValueError("Unknown primary-performance model reference")
    output.answer.tables["paper_primary_reference"] = table(
        frame.loc[frame.model.isin(models)].reset_index(drop=True),
        "Main-text means and SD, not SupplementS4 or recomputed validation. Raw percent FPR and its fraction conversion are both retained.",
        "paper_reported",
    )
    comparison = computed_primary_comparison(output, frame.loc[frame.model.isin(models)])
    if comparison:
        output.answer.tables["computed_primary_comparison"] = table(pd.DataFrame(comparison),
            "Split means compared in the same units; source cohort/folds differ or remain unverified. Differences are descriptive, not a reproduction pass/fail threshold.")
    present = {(s.name, s.sha256) for s in output.metadata.sources}
    output.metadata.sources.extend(s for s in source.sources
        if s.name in {"paper.pdf", "paper_primary_validation.csv"} and (s.name, s.sha256) not in present)
    output.answer.limitations.append(EvidenceNote(
        code="PRIMARY_REFERENCE_NOT_PAIRED_REPLAY",
        message="Main-text means and SD are separately reported references. Section4.7 describes aggregation across CV folds; "
        "the exact author partitions/checkpoints are unavailable. Do not relabel paired five-fold OOF values, a new source repeated-holdout "
        "mean, or SupplementS4 values as this primary result. The reported standardized reference total3403 is not proof of the authors' "
        "exact DMPNN training membership; this source implementation uses3171 labeled records.",
    ))
    return output
