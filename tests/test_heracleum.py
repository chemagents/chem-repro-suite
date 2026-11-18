import pytest
from fastmcp import Client

from chem_repro.heracleum import analysis
from chem_repro.heracleum.docking import protocol_validation
from chem_repro.heracleum.server import mcp


def metrics(result):
    return {m.name: m.value for m in result.answer.metrics}


def test_source_counts_and_identity_conflicts():
    m = metrics(analysis.dataset_audit())
    assert (m["ds1"], m["ds2"], m["union"], m["overlap"]) == (2008, 102, 2073, 37)
    assert m["valid_docking"] == 2055
    assert m["structure_identity_conflicts"] == 65
    assert m["generic_scaffold_conflicts"] == 2
    assert m["master_source_murcko_conflicts"] == 65
    assert m["parsable_master_smiles"] == 2073
    assert m["unusable_archived_docking_profiles"] == 18
    r = analysis.dataset_audit()
    rows = r.answer.tables["unusable_archived_profiles"].rows
    assert len(rows) == 18 and all(row["all_44_archived_scores_zero"] for row in rows)
    assert "ARCHIVED_PROFILE_NOT_GEOMETRY_VALIDATION" in {note.code for note in r.answer.limitations}
    identity_rows = r.answer.tables["identity_audit"].rows
    assert all(not row["master_source_murcko_matches"] for row in identity_rows)
    assert sum(not row["generic_scaffold_matches"] for row in identity_rows) == 2


def test_scaffold_concentration_and_singleton_denominators():
    r = analysis.chemical_space()
    m = metrics(r)
    assert m["DS1_psoralen_scaffold_n"] == 767 and m["DS1_angelicin_scaffold_n"] == 334
    rows = {v["subset"]: v for v in r.answer.tables["diversity"].rows}
    assert rows["DS1"]["top5_n"] == 1254
    assert rows["DS2"]["top15_n"] == 79
    # Article's 23 describes the remainder after top-15, not all singleton scaffolds.
    assert rows["DS2"]["singleton_scaffolds_total"] == 28
    assert rows["DS2"]["singleton_scaffolds_outside_top15"] == 23


def test_pca_numbers():
    d = analysis.docking_profile()
    t = analysis.toxicity_landscape()
    assert metrics(d)["components_for_80_percent"] == 4
    assert metrics(d)["pc1_mean_docking_spearman"] == pytest.approx(0.927, abs=0.002)
    assert metrics(t)["components_for_80_percent"] == 6
    variance = t.answer.tables["pca_variance"].rows
    assert variance[4]["cumulative_fraction"] < 0.8 <= variance[5]["cumulative_fraction"]
    assert variance[5]["cumulative_fraction"] == pytest.approx(0.82118453275)
    assert "at least 80%" in t.answer.finding
    statistics = {r["endpoint"]: r for r in t.answer.tables["endpoint_statistics"].rows}
    assert statistics["genotoxicity"]["iqr_rank_ascending"] == 1
    assert statistics["ototoxicity"]["iqr_rank_ascending"] > 1
    assert d.answer.tables["pca_variance"].rows[0]["variance_fraction"] == pytest.approx(0.612, abs=0.003)


def test_tiering_cannot_hide_conflicts():
    raw = analysis.prioritize("published_snapshot")
    strict = analysis.prioritize("strict")
    assert metrics(raw) == {"tier_1": 21, "tier_2": 24, "tier_3": 57}
    assert metrics(strict) == {"tier_1": 18, "tier_2": 10, "tier_3": 9, "tier_unranked": 65}
    for row in strict.answer.tables["candidates"].rows:
        if not row["identity_verified"]:
            assert row["tier"] is None and row["name"] is None


def test_alerts_follow_distinct_structure_sources_without_replacing_tier_inputs():
    from chem_repro.heracleum.alerts import alert_audit

    rows = alert_audit()
    assert len(rows) == 102
    assert rows.original_DS2_counts_match_archive.all()
    assert int(rows.master_counts_match_archive.sum()) == 50
    assert int((rows.master_pains_count > 0).sum()) == 3
    assert int((rows.original_DS2_pains_count > 0).sum()) == 0
    assert not rows.loc[rows.master_pains_count > 0, "identity_verified"].any()


def test_ks_effects_and_fdr_resolution():
    r = analysis.separability(99)
    rows = {v["axis"]: v for v in r.answer.tables["axis_tests"].rows}
    assert rows["PC1"]["observed_ks"] == pytest.approx(0.832, abs=0.003)
    assert rows["neurotoxicity"]["observed_ks"] == pytest.approx(0.509, abs=0.003)
    assert all(v["p_permutation"] >= 0.01 for v in rows.values())


def test_elasticnet_cv_and_fold_evidence():
    r = analysis.transfer("neurotoxicity", "ElasticNet", "published_snapshot", 100, True)
    rows = {v["scheme"]: v for v in r.answer.tables["validation_metrics"].rows}
    assert rows["random"]["r2"] == pytest.approx(0.310015, abs=0.002)
    assert rows["scaffold"]["r2"] == pytest.approx(0.195461, abs=0.002)
    assert all(
        v["generic_scaffold_overlap"] == 0
        for v in r.answer.tables["fold_audit"].rows
        if v["scheme"] == "scaffold"
    )
    assert len(r.answer.tables["oof_predictions"].rows) == 2073 * 2


async def test_heracleum_tool_contract():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert len(tools) == 8
        assert all(t.inputSchema["x-data-contract"]["dataset_scope"] == "heracleum_2026" for t in tools)
        transfer_tool = next(t for t in tools if t.name == "heracleum2026_scaffold_transfer")
        assert transfer_tool.inputSchema["properties"]["identity_policy"]["default"] == "strict"
        r = await client.call_tool("heracleum2026_chemical_space")
        assert not r.is_error
        assert r.structured_content["metadata"]["study_id"] == "heracleum_2026"


def test_redocking_aggregates_published_observations_without_claiming_execution():
    r = protocol_validation()
    m = metrics(r)
    assert m["evaluated_targets"] == 42 and m["excluded_targets"] == 2
    assert m["median_best_top5_rmsd"] == pytest.approx(0.83)
    rows = r.answer.tables["pose_recovery"].rows
    assert [row["n_recovered"] for row in rows] == [24, 31, 35, 36, 36]
    assert {row["target"] for row in r.answer.tables["failed_targets"].rows} == {
        "HRH2",
        "HTR1A",
        "CACNA1C",
        "SCN5A",
        "PTGS1",
        "MAOA",
    }
    assert r.status == "partial"
    assert r.answer.tables["target_observations"].evidence_kind == "source_record"
    source = next(s for s in r.metadata.sources if s.name == "redocking_table_s1a.csv")
    parent = next(s for s in r.metadata.sources if s.name == "supplement.pdf")
    assert source.parent_sha256 == parent.sha256 and source.derivation
    assert "MDPI supplement" in source.revision
