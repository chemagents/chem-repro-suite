import json

import pytest
from fastmcp import Client

from chem_repro.mannich import analysis
from chem_repro.mannich.server import mcp


def metric_map(result):
    return {m.name: m.value for m in result.answer.metrics}


def test_dataset_counts_and_axial_stereochemistry():
    r = metric_map(analysis.overview())
    assert r["reaction_records"] == 4015
    assert r["source_publications"] == 153
    assert r["raw_doi_strings"] == 154
    assert r["unique_catalysts_including_axial"] == 468
    assert r["unique_catalyst_isomeric_smiles"] == 462
    assert r["unique_nucleophiles"] == 538
    assert r["unique_electrophiles"] == 578


def test_eyring_recomputed_and_publication_distribution():
    r = analysis.target_distribution()
    assert metric_map(r)["ddg_max_absolute_residual"] < 1e-12
    rows = {x["target"]: x for x in r.answer.tables["target_statistics"].rows}
    assert rows["ee_percent"]["median"] == 89.4
    assert rows["ee_percent"]["skewness"] == pytest.approx(-1.55, abs=0.005)
    assert rows["recomputed_abs_ddg_kcal_mol"]["skewness"] == pytest.approx(0.10, abs=0.005)


def test_conditions_have_expected_units_and_medians():
    r = analysis.conditions()
    rows = {x["field"]: x for x in r.answer.tables["condition_statistics"].rows}
    assert rows["temperature"]["min"] == -78 and rows["temperature"]["max"] == 85
    assert rows["yield"]["median"] == 83
    assert rows["time"]["median"] == 24


def test_report_statistics_keep_estimators_and_denominators_explicit():
    r = analysis.target_distribution()
    row = r.answer.tables["target_statistics"].rows[0]
    assert row["skewness_bias_corrected"] != row["skewness"]
    assert round(row["skewness_bias_corrected"], 2) == round(row["skewness"], 2) == -1.55
    r = analysis.component_classes()
    dominant = r.answer.tables["dominant_catalyst_classes"].rows
    assert [x["n_reactions"] for x in dominant] == [1375, 895, 617, 592]
    assert sum(x["percent_of_all_reactions"] for x in dominant) == pytest.approx(3479 / 4015 * 100)
    assert metric_map(analysis.overview(False))["source_fields"] == 27


def test_scaffold_membership_can_reconstruct_all_counts_without_dropping_acyclics():
    r = analysis.scaffold_diversity()
    members = r.answer.tables["scaffold_assignments"].rows
    assert len(members) == 468 + 578 + 538
    for row in r.answer.tables["role_summary"].rows:
        subset = [m for m in members if m["role"] == row["role"]]
        assert len(subset) == row["unique_structures"]
        assert sum(m["acyclic_excluded_from_scaffold_count"] for m in subset) == row["acyclic_excluded"]
        assert len({m["scaffold_smiles"] for m in subset if m["scaffold_smiles"]}) == row["nonempty_scaffolds"]


def test_supplement_scaffolds_and_acyclic_exclusions():
    r = analysis.scaffold_diversity()
    assert r.status == "partial"
    rows = {x["role"]: x for x in r.answer.tables["role_summary"].rows}
    # The declared paper method gives 207 on both RDKit 2025.09.2 and 2026.03.3.
    # The test requires the unresolved disagreement to remain visible, not a fabricated match.
    assert rows["organocatalyst"]["nonempty_scaffolds"] == 207
    assert rows["organocatalyst"]["acyclic_excluded"] == 16
    assert rows["electrophile"]["nonempty_scaffolds"] == 114
    assert rows["nucleophile"]["nonempty_scaffolds"] == 112
    comparison = r.answer.tables["reference_comparison"].rows[0]
    assert comparison["paper_reported_scaffolds"] == 204
    assert comparison["matches"] is False
    assert any(n.code == "SUPPLEMENT_SCAFFOLD_DISAGREEMENT" for n in r.answer.limitations)
    audit = r.answer.tables["software_version_check"]
    assert audit.evidence_kind == "source_record" and len(audit.rows) == 12
    assert {row['rdkit'] for row in audit.rows} == {'2025.09.2','2025.09.6','2026.03.3'}
    assert all(row['source_records'] == 4015 for row in audit.rows)
    assert all(row['individual_scaffolds_match_reference_version'] for row in audit.rows)
    assert {row['paper_method_scaffolds'] for row in audit.rows if row['role']=='organocatalyst'} == {207}
    assert r.metadata.protocol_parameters['version_diagnostic_source'] == 'stored_own_reconstruction'


def test_smarts_differentiates_reactive_families():
    assert analysis.classify_molecule("C[N+](=O)[O-]", "nucleophile") == "nitroalkane"
    assert analysis.classify_molecule("CC(=O)CC(=O)OC", "nucleophile") == "beta_ketoester"
    assert analysis.classify_molecule("CC(=O)O", "nucleophile") == "other"
    r = analysis.component_classes()
    assert len(r.answer.tables["class_frequencies"].rows) > 10
    counts = {row["role"]: row for row in r.answer.tables["class_count_summary"].rows}
    assert [counts[role]["classes_including_other"] for role in ["organocatalyst", "nucleophile", "electrophile"]] == [11, 19, 10]
    assert all(row["reaction_records"] == 4015 for row in counts.values())


def test_heatmap_protocol_difference_is_preserved():
    paper = analysis.selectivity_heatmaps("paper")
    notebook = analysis.selectivity_heatmaps("notebook")
    assert paper.metadata.protocol_parameters["minimum_cell_count"] == 5
    assert notebook.metadata.protocol_parameters["minimum_cell_count"] == 4
    for role in ("electrophile", "nucleophile"):
        table = paper.answer.tables[role + "_heatmap"]
        assert all(x["display_median_ee"] is None for x in table.rows if x["n_reactions"] < 5)
        assert all(x["n_reactions"] >= 5 for x in paper.answer.tables[role + "_visible_by_median"].rows)


async def test_mcp_declared_output_and_dataset_scope():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert len(tools) == 10
        for tool in tools:
            assert tool.inputSchema["x-data-contract"] == {
                "input_mode": "fixed_dataset",
                "dataset_scope": "mannich_2026",
                "optional_numeric_defaults_are_operational": True,
            }
            assert "answer" in tool.outputSchema["properties"]
            if tool.name not in {
                "mannich_dataset_overview",
                "mannich_enantioselectivity_distribution",
                "mannich_reaction_conditions",
                "mannich_chemical_space",
                "mannich_selectivity_heatmaps",
                "mannich_component_classification",
                "mannich_scaffold_diversity",
                "mannich_reaction_space",
                "mannich_regenerate_catalyst_structures",
            }:
                assert tool.annotations.readOnlyHint
        result = await client.call_tool("mannich_dataset_overview")
        assert not result.is_error
        json.dumps(result.structured_content, allow_nan=False)
        assert result.structured_content["metadata"]["software"]["rdkit"]
