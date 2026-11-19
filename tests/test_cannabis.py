import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

from chem_repro.cannabis import analysis
from chem_repro.cannabis.data import TARGETS, split
from chem_repro.cannabis.server import mcp
from chem_repro.common.rmt import fit_residue_selector


def fixture_energies():
    rng = np.random.default_rng(12)
    y = np.tile([0, 1], 70)
    latent = -1.5 * y + rng.normal(0, 0.5, len(y))
    x = latent[:, None] * np.arange(1, 9) + rng.normal(0, 0.3, (len(y), 8))
    return x, y, [f"chain_A_residue_{i}" for i in range(8)]


def test_selector_preserves_raw_energies_and_freezes_training_selection():
    x, y, names = fixture_energies()
    model = fit_residue_selector(x, y, names, inner_splits=2)
    before = model.ranking.copy(deep=True)
    selected = model.selected_indices.copy()
    heldout = np.arange(24, dtype=float).reshape(3, 8)
    assert np.array_equal(model.transform(heldout, names), heldout[:, selected])
    model.transform(heldout * 1e6, names)
    pd.testing.assert_frame_equal(before, model.ranking)
    assert np.array_equal(selected, model.selected_indices)
    with pytest.raises(ValueError, match="names and order"):
        model.transform(heldout, names[::-1])
    r = model.ranking.loc[model.ranking.in_top_m].sort_values("energy_rank_within_top_m")
    kept = r.loc[r.selected]
    assert kept.cumulative_contribution_fraction.iloc[-1] >= 0.98
    if len(kept) > 1:
        assert kept.cumulative_contribution_fraction.iloc[-2] < 0.98


def test_selector_validates_degenerate_energy_gate():
    x, y, names = fixture_energies()
    with pytest.raises(ValueError, match="zero class-mean"):
        fit_residue_selector(np.zeros_like(x), y, names, inner_splits=2)
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        fit_residue_selector(x, y, names, inner_splits=2)


def test_cannabis_cohorts_are_not_silently_relabelled():
    r = analysis.dataset_audit()
    m = {v.name: v.value for v in r.answer.metrics}
    assert m == {"source_records": 5920, "labeled_records": 3171, "unlabeled_records": 2749}
    assert r.status == "partial"
    assert "SOURCE_COHORT_MISMATCH" in {n.code for n in r.answer.limitations}
    flow = r.answer.tables["paper_cohort_flow"]
    assert flow.evidence_kind == "paper_reported"
    assert [row["paper_n"] for row in flow.rows if row["paper_cohort"] == "DS1_Cannabis"] == [5211, 5027, 2465]
    train, test = split(0)
    assert len(train) == 2536 and len(test) == 635
    assert not set(train.ligand_id) & set(test.ligand_id)


def test_source_overlap_reports_identities_and_records_separately():
    rows = analysis.dataset_audit().answer.tables["cross_cohort_identity_overlap"].rows
    exact = {
        (r["left_cohort"], r["right_cohort"]): r for r in rows if r["identity_policy"] == "exact_isomeric"
    }
    positive = exact[("DS2_pesticides", "source_unlabeled")]
    negative = exact[("DS3_AChE_inactive", "source_unlabeled")]
    assert positive["shared_identities"] + negative["shared_identities"] == 384
    assert positive["right_records"] + negative["right_records"] == 464
    assert exact[("DS2_pesticides", "DS3_AChE_inactive")]["shared_identities"] == 0


def test_six_target_comparisons_include_opposite_direction():
    result = analysis.docking_comparisons()
    assert result.status == "success"
    assert result.metadata.protocol_parameters["unavailable_ds1_map_affects_this_comparison"] is False
    rows = {r["target"]: r for r in result.answer.tables["target_tests"].rows}
    assert set(rows) == set(TARGETS)
    assert rows["OR28"]["median_difference_DS2_minus_DS3"] > 0
    assert sum(r["median_difference_DS2_minus_DS3"] < 0 for r in rows.values()) == 5
    assert all(r["n_DS2"] == 1680 and r["n_DS3"] == 1491 for r in rows.values())
    assert all(r["p_fdr"] >= r["p_two_sided"] for r in rows.values())


def test_selected_residue_labels_match_exported_energy_columns(tmp_path, monkeypatch):
    import json

    from chem_repro.common.artifacts import export_tables

    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR", str(tmp_path))
    result = analysis.select_residues("GSTE2", inner_splits=5)
    selected = result.answer.tables["selected_residues"]
    labels = [row["residue"] for row in selected.rows]
    assert len(labels) == 60 and len(set(labels)) == 60
    original = next(a for a in result.artifacts if a.name.endswith(".json"))
    payload = json.loads((tmp_path / (original.sha256 + ".json")).read_text())
    for key in ["train_selected_energies", "test_selected_energies"]:
        assert [name for name in payload[key][0] if name != "ligand_id"] == labels
    result = export_tables(result, port=7341)
    assert not result.answer.tables["selected_residues"].is_preview
    assert [r["residue"] for r in result.answer.tables["selected_residues"].rows] == labels


async def test_cannabis_transport_contract():
    async with Client(mcp) as client:
        listed = await client.list_tools()
        assert len(listed) == 11
        assert {'cannabis2026_reproduce_author_code', 'cannabis2026_recovered_author_cohort'} <= {t.name for t in listed}
        for tool in listed:
            scope = (
                "cannabis_open_tox_2026"
                if tool.name == "cannabis2026_open_toxicity_validation"
                else "cannabis_2026"
            )
            assert tool.inputSchema["x-data-contract"]["dataset_scope"] == scope
        response = await client.call_tool("cannabis2026_docking_comparisons")
        assert not response.is_error
        assert len(response.structured_content["answer"]["tables"]["target_tests"]["rows"]) == 6


def test_all_target_diagnostic_preserves_each_target_and_same_source_split(monkeypatch):
    from chem_repro.cannabis import server
    received=[]
    def select(target, index, inner_splits, test_size, seed):
        received.append((target,index,inner_splits,test_size,seed))
        return analysis.result("verified " + target,[],{},"test", target=target,split_index=index)
    monkeypatch.setattr(analysis,"select_residues",select)
    monkeypatch.setattr("chem_repro.cannabis.figures.selection",lambda r:r)
    output=server.select_residues(target="all",split_index=2)
    assert [v[0] for v in received]==list(TARGETS)
    assert all(v[1:]==(2,5,.2,42) for v in received)
    assert len(output.answer.tables["variant_inventory"].rows)==6
def test_docking_mean_and_median_are_not_silently_interchanged():
    from chem_repro.cannabis.analysis import docking_comparisons
    result=docking_comparisons()
    rows={r['target']:r for r in result.answer.tables['target_tests'].rows}
    assert round(rows['OR28']['mean_difference_DS2_minus_DS3'],2)==1.20
    assert round(rows['OR28']['median_difference_DS2_minus_DS3'],2)==1.40
    means=[r['mean_difference_DS2_minus_DS3'] for t,r in rows.items() if t!='OR28']
    assert round(min(means),2)==-1.01 and round(max(means),2)==-.70
    assert any(n.code=='PAPER_MEAN_MEDIAN_DISCREPANCY' for n in result.answer.limitations)

