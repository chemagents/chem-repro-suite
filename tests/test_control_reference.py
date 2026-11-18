import hashlib

from chem_repro.heracleum.control_reference import attach_control_reference
from chem_repro.heracleum.docking import protocol_validation


def test_control_reference_remains_separate_from_recomputed_redocking(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR", str(tmp_path))
    original = protocol_validation()
    recovery = original.answer.tables["pose_recovery"].model_dump()
    result = attach_control_reference(original)
    assert result.answer.tables["pose_recovery"].model_dump() == recovery
    metrics = {m.name: m for m in result.answer.metrics}
    assert metrics["evaluated_targets"].value == 42
    assert metrics["paper_targets_with_sufficient_observations"].value == 39
    assert metrics["paper_targets_active_better_significant"].value == 13
    assert metrics["paper_inactive_control_panel_median_kcal_mol"].value == -7.3
    assert metrics["paper_active_control_panel_median_kcal_mol"].value == -7.8
    assert all(m.evidence_kind == "paper_reported" for n, m in metrics.items() if n.startswith("paper_"))
    tab = result.answer.tables["paper_reported_control_discrimination"]
    assert tab.evidence_kind == "paper_reported" and tab.rows[0]["multiplicity_correction"] == "Bonferroni"
    assert tab.rows[0]["alpha"] == 0.05 and tab.rows[0]["test_alternative"] == "one-sided"
    assert tab.rows[0]["control_database"] == "PubChem BioAssay"
    assert tab.rows[0]["per_target_pvalues_available"] is False
    assert result.metadata.protocol_parameters["control_reference"]["original_control_test_recomputed"] is False
    parent = next(s for s in result.metadata.sources if s.name == "paper.pdf")
    derived = next(s for s in result.metadata.sources if s.name == "control_reference.json")
    pdf = next(a for a in result.artifacts if a.name == "heracleum_paper_source_reference.pdf")
    assert pdf.sha256 == parent.sha256 == derived.parent_sha256
    assert hashlib.sha256((tmp_path / (pdf.sha256 + ".pdf")).read_bytes()).hexdigest() == parent.sha256
    n_metrics, n_artifacts = len(result.answer.metrics), len(result.artifacts)
    attach_control_reference(result)
    assert (len(result.answer.metrics), len(result.artifacts)) == (n_metrics, n_artifacts)
