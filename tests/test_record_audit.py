import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest

from chem_repro.common.alerts import structural_alerts
from chem_repro.common.artifacts import artifact_root
from chem_repro.common.record_audit import heavy_atom_balance, parse_reported_ratio, structure_diagnostics
from chem_repro.mannich import analysis


def test_dr_ratio_keeps_censoring_and_zero_minor_component():
    assert parse_reported_ratio("19:1")["reported_fractions"] == [0.95, 0.05]
    assert parse_reported_ratio(">19:1")["reported_fractions"] is None
    assert parse_reported_ratio("100:0")["reported_fractions"] == [1, 0]
    with pytest.raises(ValueError):
        parse_reported_ratio("0:0")
    with pytest.raises(ValueError):
        parse_reported_ratio("90% ee")


def test_atom_inventory_preserves_isotopes_and_separates_charge():
    assert heavy_atom_balance(["CC(C)=O", "C=N"], "CC(=O)CCN")["heavy_atom_delta"] == {}
    assert heavy_atom_balance(["[13CH4]"], "C")["heavy_atom_delta"] == {"13C": -1, "C": 1}
    assert structure_diagnostics("CCO.[Na+]")["fragments"] == 2
    assert structure_diagnostics("CCO.[Na+]")["formal_charge"] == 1
    assert structure_diagnostics("NC(C)C(=O)O")["unassigned_carbon_stereocenters"]
    assert not structure_diagnostics("N[C@H](C)C(=O)O")["unassigned_carbon_stereocenters"]


def test_full_record_audit_keeps_all_rows_and_high_additive_loadings():
    result = analysis.overview()
    assert all(r["n_invalid"] == 0 for r in result.answer.tables["numeric_field_audit"].rows)
    base = next(
        r
        for r in result.answer.tables["numeric_field_audit"].rows
        if r["field"] == "base_additive_concentration(mol%)"
    )
    assert base["max_observed"] == 500 and base["upper"] is None
    flags = {r["flag"]: r["n_records"] for r in result.answer.tables["quality_flag_counts"].rows}
    assert flags["base_additive_loading_unreported"] == 1
    assert flags["product_unspecified_carbon_stereo"] == 20
    file = artifact_root() / Path(urlparse(result.artifacts[0].download_url).path).name
    full = pd.read_csv(file)
    assert len(full) == 4015 and full.reaction_id.nunique() == 4015
    assert json.loads(full.iloc[0].role_fragment_counts)["product"] == 1
    assert full.source_row_1based.tolist() == list(range(1, 4016))


def test_catalog_counts_are_alert_entries_not_safety_predictions():
    assert structural_alerts("CCO")["pains_count"] == 0
    positive = structural_alerts("O=C(Cn1cnc2c1c(=O)n(C)c(=O)n2C)N/N=C/c1c(O)ccc2c1cccc2")
    assert any("hzone_phenol" in name for name in positive["pains_names"])
    assert positive["pains_count"] == len(positive["pains_names"])


def test_solvent_label_count_is_not_a_molecular_identity_claim():
    result = analysis.conditions()
    metrics = {m.name: m.value for m in result.answer.metrics}
    assert metrics["nonmixture_solvent_labels_excluding_neat"] == 51
    assert metrics["normalized_solvent_categories"] == 50
    assert any(n.code == "SOLVENT_COUNT_DEFINITIONS" for n in result.answer.limitations)


def test_repeated_group_and_record_counts_have_separate_units():
    result=analysis.overview()
    metrics={m.name:m for m in result.answer.metrics}
    groups=result.answer.tables["repeated_input_groups"].rows
    assert len(groups)==metrics["repeated_input_groups_with_different_outcomes"].value==46
    assert sum(row["n_records"] for row in groups)==metrics["records_in_repeated_input_groups"].value==105
    assert metrics["repeated_input_groups_with_different_outcomes"].unit=="groups"
    assert metrics["records_in_repeated_input_groups"].unit=="records"
    assert result.metadata.paper_doi in result.answer.finding
