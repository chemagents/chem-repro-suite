import copy

import numpy as np
import pandas as pd
import pytest

from chem_repro.common.classification import evaluate_product
from chem_repro.common.matched_validation import summarize_matched_predictions


BRANCHES = {"structure": "structure_probability", "rte": "rte_probability", "product": "product_score"}


def fixture():
    rows, metrics = [], []
    for fold, ids, a, b in [
        (0, [1, 2, 3, 4], [.2, .6, .8, .4], [.5, .7, .9, .3]),
        (1, [1, 2, 5, 6], [.1, .2, .9, .6], [.2, .8, .8, .6]),
    ]:
        y = [0, 0, 1, 1]
        evaluated = evaluate_product(y, a, b)
        for branch in BRANCHES:
            metrics.append({"scheme": "repeat", "scope": "fold", "fold": fold, "branch": branch,
                            **evaluated[branch]})
        rows.extend({"fold": fold, "ligand_id": identity, "activity": label,
                     "structure_probability": x, "rte_probability": z, "product_score": x*z}
                    for identity, label, x, z in zip(ids, y, a, b))
    return {"repeat": pd.DataFrame(rows)}, metrics


def summarize(predictions, metrics):
    return summarize_matched_predictions(predictions, metrics, branches=BRANCHES,
                                         candidate="product", source_ids=range(1, 9))


def test_repeated_holdout_summary_separates_instances_unique_records_and_retention():
    result = summarize(*fixture())
    coverage = result["scoring_coverage"].iloc[0]
    assert coverage.matched_scored_test_instances == coverage.declared_test_instances == 8
    assert coverage.test_instance_coverage == 1
    assert coverage.unique_test_records == 6 and coverage.source_record_coverage == .75
    summary = result["fold_summary"].set_index("branch")
    assert summary.loc["structure", "mean_roc_auc"] == .875
    assert summary.loc["structure", "mean_fpr"] == .25
    assert summary.loc["structure", "sd_fpr"] == pytest.approx(np.sqrt(.125))
    assert summary.loc["product", "mean_recall"] == .5
    assert summary.loc["product", "sum_fp"] == 0
    assert summary.loc["product", "sum_n_test"] == 8  # six unique IDs, eight predictions
    differences = result["paired_summary"].set_index("reference")
    assert set(differences.index) == {"structure", "rte"}
    assert differences.loc["structure", "mean_delta_fpr"] == -.25
    assert differences.loc["structure", "mean_delta_recall"] == -.25
    assert differences.loc["structure", "relative_fpr_reduction_of_means"] == 1


@pytest.mark.parametrize("problem", ["missing_branch", "duplicate_id", "missing_row", "wrong_metric", "fractional_label"])
def test_incomplete_or_inconsistent_evidence_is_not_summarized(problem):
    predictions, metrics = fixture()
    metrics = copy.deepcopy(metrics)
    if problem == "missing_branch":
        metrics.pop()
    elif problem == "duplicate_id":
        predictions["repeat"] = pd.concat([predictions["repeat"], predictions["repeat"].iloc[:1]])
    elif problem == "missing_row":
        predictions["repeat"] = predictions["repeat"].iloc[1:]
    elif problem == "wrong_metric":
        metrics[0]["roc_auc"] = .123
    else:
        predictions["repeat"]["activity"] = predictions["repeat"].activity.astype(float)
        predictions["repeat"].loc[0, "activity"] = .5
    with pytest.raises(ValueError):
        summarize(predictions, metrics)


def test_single_fold_does_not_invent_an_sd():
    predictions, metrics = fixture()
    predictions["repeat"] = predictions["repeat"].loc[lambda x:x.fold == 0]
    result = summarize(predictions, [r for r in metrics if r["fold"] == 0])
    assert result["fold_summary"].sd_roc_auc.isna().all()


@pytest.mark.parametrize("corrupt", [False, True])
def test_result_adapter_reads_complete_artifacts_instead_of_previews(tmp_path, monkeypatch, corrupt):
    from chem_repro.common.artifacts import export_tables, save_artifact
    from chem_repro.common.contracts import AnalysisResult, Answer
    from chem_repro.common.datasets import provenance, table
    from chem_repro.cannabis import fusion_summary

    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR", str(tmp_path))
    predictions, metrics = fixture()
    source = pd.DataFrame({"ligand_id": range(1, 9), "activity": [0, 0, 1, 1, 1, 1, 1, 1]})
    monkeypatch.setattr(fusion_summary, "load", lambda: source)
    output = AnalysisResult(status="success", metadata=provenance("cannabis", "test.matched"),
                            answer=Answer(finding="Synthetic test fixture", tables={
                                "paired_metrics": table(pd.DataFrame(metrics), "Synthetic complete metrics")}))
    artifact = save_artifact(predictions["repeat"].to_csv(index=False).encode(), suffix="csv",
                             media_type="text/csv", public_base_url="http://127.0.0.1:7341",
                             display_name="repeat_fusion_heldout_predictions.csv")
    output.artifacts.append(artifact)
    output = export_tables(output, port=7341, preview_rows=1, compact_max_rows=0)
    assert len(output.answer.tables["paired_metrics"].rows) == 1
    if corrupt:
        (tmp_path / (artifact.sha256 + ".csv")).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            fusion_summary.attach_fusion_summary(output)
    else:
        completed = fusion_summary.attach_fusion_summary(output)
        assert len(completed.answer.tables["fold_summary"].rows) == 3
        assert completed.answer.tables["scoring_coverage"].rows[0]["unique_test_records"] == 6
        assert completed.answer.tables["paired_metrics"].is_preview
