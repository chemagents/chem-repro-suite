import pytest

from chem_repro.cannabis.paper_reference import attach_primary_validation
from chem_repro.common.contracts import AnalysisResult, Answer, Metric, Provenance


def output():
    return AnalysisResult(status="partial", answer=Answer(finding="Computed reconstruction", metrics=[
        Metric(name="computed_auc", value=.91, population="source OOF")]),
        metadata=Provenance(study_id="cannabis_2026", paper_doi="10.3390/ijms27188000", protocol_id="test", sources=[], software={}))


def test_primary_reference_is_not_a_computed_or_supplement_metric():
    result = output()
    before = result.answer.metrics[0].model_dump()
    attach_primary_validation(result, {"DMPNN", "product"})
    data = result.answer.tables["paper_primary_reference"]
    assert data.evidence_kind == "paper_reported" and len(data.rows) == 4
    assert result.answer.metrics[0].model_dump() == before
    rows = {(r["model"], r["metric"]): r for r in data.rows}
    assert rows[("DMPNN", "ROC_AUC")]["reported_mean"] == .9283
    assert rows[("DMPNN", "ROC_AUC")]["reported_sd"] == .0092
    fpr = rows[("DMPNN", "FPR")]
    assert fpr["reported_mean"] == 12.2 and fpr["reported_unit"] == "percent"
    assert fpr["comparison_mean"] == .122 and fpr["comparison_sd"] == .0186
    assert fpr["evaluation_threshold"] == .5
    assert rows[("product", "FPR")]["comparison_mean"] == .0492
    assert result.status == "partial"


def test_primary_reference_rejects_unknown_model_and_deduplicates_sources():
    result = output()
    with pytest.raises(ValueError):
        attach_primary_validation(result, {"unknown"})
    attach_primary_validation(result, {"RTE_ALL6"})
    attach_primary_validation(result, {"RTE_ALL6"})
    assert len(result.metadata.sources) == 2
    assert result.answer.tables["paper_primary_reference"].rows[0]["reported_mean"] == .802
