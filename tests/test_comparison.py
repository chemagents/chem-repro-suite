import pytest
from chem_repro.common.comparison import compare_variants
from chem_repro.common.contracts import AnalysisResult, Answer, Metric, Provenance, Table
from chem_repro.mannich import analysis
from chem_repro.mannich import server


def result(value, status="success"):
    return AnalysisResult(status=status, answer=Answer(finding="computed", metrics=[Metric(name="n", value=value, population="records")],
        tables={"rows": Table(description="records", columns=["id"], rows=[{"id":value}])}),
        metadata=Provenance(study_id="study", paper_doi="10.1/test", protocol_id=str(value), sources=[], software={"test":"1"}))


def test_comparison_preserves_variant_values_statuses_and_originals():
    first, second = result(3), result(7, "partial")
    before = first.model_dump()
    combined = compare_variants({"paper":first,"notebook":second}, protocol_id="comparison", finding="comparison")
    assert combined.status == "partial"
    assert {m.name:m.value for m in combined.answer.metrics} == {"paper__n":3,"notebook__n":7}
    assert combined.answer.tables["paper__rows"].rows == [{"id":3}]
    assert combined.answer.tables["notebook__rows"].rows == [{"id":7}]
    combined.answer.tables["paper__rows"].rows[0]["id"] = 99
    assert first.model_dump() == before
    assert combined.metadata.protocol_parameters["variants"]["notebook"]["protocol_id"] == "7"


def test_comparison_rejects_mixed_studies_or_implementations():
    a,b=result(3),result(7)
    b.metadata.study_id="different"
    with pytest.raises(ValueError,match="same study"):compare_variants({"a":a,"b":b},protocol_id="c",finding="c")
    b=result(7);b.metadata.implementation_sha256="0"*64
    with pytest.raises(ValueError,match="same study"):compare_variants({"a":a,"b":b},protocol_id="c",finding="c")


def test_heatmap_both_runs_the_two_real_protocols_without_population_changes():
    both=server.selectivity_heatmaps("both",include_figures=False)
    assert both.status=="success"
    for mode,minimum in [("paper",5),("notebook",4)]:
        assert both.metadata.protocol_parameters["variants"][mode]["protocol_parameters"]["minimum_cell_count"]==minimum
        for role in ["electrophile","nucleophile"]:
            rows=both.answer.tables[f"{mode}__{role}_heatmap"].rows
            assert rows == analysis.selectivity_heatmaps(mode).answer.tables[role+"_heatmap"].rows
    for role in ("electrophile", "nucleophile"):
        paper = both.answer.tables[f"paper__{role}_heatmap"].rows
        notebook = both.answer.tables[f"notebook__{role}_heatmap"].rows
        for left, right in zip(paper, notebook):
            assert left["n_reactions"] == right["n_reactions"]
            assert (left["visible"] != right["visible"]) == (left["n_reactions"] == 4)



def test_full_molecular_space_request_is_one_labeled_three_variant_result(monkeypatch):
    from chem_repro.mannich import space
    calls=[]
    def compute(pop,mode):
        calls.append((pop,mode));r=result(len(calls));r.metadata.protocol_id=pop+"."+mode;return r
    monkeypatch.setattr(space,"chemical_space",compute)
    combined=server.chemical_space("both","compare",include_figures=False)
    assert calls==[("catalysts","notebook"),("catalysts","chirality_sensitivity"),("reagents","notebook")]
    assert len(combined.answer.tables["variant_inventory"].rows)==3
    assert set(combined.metadata.protocol_parameters["variants"])=={"catalysts_notebook","catalysts_chirality_sensitivity","reagents_notebook"}


def test_every_eyring_record_is_exported_and_zero_references_are_not_divided():
    r=analysis.target_distribution();metrics={m.name:m.value for m in r.answer.metrics}
    rows=r.answer.tables["conversion_records"].rows
    assert len(rows)==metrics["recomputed_records"]==4015
    assert len({row["reaction_id"] for row in rows})==4015
    zeros=[row for row in rows if row["ddG_kcal_mol"]==0]
    assert len(zeros)==metrics["zero_ddg_records"]==43
    assert all(row["absolute_relative_error_percent"] is None and row["recomputed_ddg"]==0 for row in zeros)
    assert metrics["nonzero_ddg_records_for_relative_error"]==3972
    assert metrics["median_absolute_relative_error_percent"]==0
    assert not r.answer.tables["conversion_disagreements"].rows
    assert any(n.code=="EE_VERSUS_DR" for n in r.answer.limitations)


def test_identity_comparison_keeps_conflicts_and_both_ranking_populations(monkeypatch,tmp_path):
    from chem_repro.heracleum.server import prioritize_candidates
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR",str(tmp_path))
    combined=prioritize_candidates("compare",include_figures=False)
    for policy in ("strict","published_snapshot"):
        direct=prioritize_candidates(policy,include_figures=False)
        for metric in direct.answer.metrics:
            value=next(m.value for m in combined.answer.metrics if m.name==policy+"__"+metric.name)
            assert value==metric.value
    assert set(combined.metadata.protocol_parameters["variants"])=={"published_snapshot","strict"}


async def test_tools_explain_direct_result_binding_without_inventing_output_files():
    from fastmcp import Client
    async with Client(server.mcp) as client:
        tools=await client.list_tools()
    for tool in tools:
        assert f"{tool.name}.json" in tool.description
        assert "ONE complete AnalysisResult JSON" in tool.description
        assert f"{tool.name}.json" in tool.outputSchema["description"]
        assert set(tool.outputSchema["properties"]) >= {"answer","metadata","artifacts"}


def test_real_mask_comparison_does_not_invent_cells_at_the_cutoff():
    r=server.selectivity_heatmaps("both",include_figures=False)
    rows={row["substrate_role"]:row for row in r.answer.tables["masking_comparison"].rows}
    assert rows["electrophile"]["observed_class_pairs"]==53
    assert rows["nucleophile"]["observed_class_pairs"]==67
    assert all(row["pairs_with_n4"]==0 and row["changed_visibility_pairs"]==0 for row in rows.values())
    assert rows["electrophile"]["paper_visible_pairs"]==rows["electrophile"]["notebook_visible_pairs"]==41
    assert rows["nucleophile"]["paper_visible_pairs"]==rows["nucleophile"]["notebook_visible_pairs"]==53
    assert r.answer.tables["changed_mask_cells"].rows==[]


def test_single_protocol_has_full_mask_counts_and_preserves_group_value_associations():
    r=analysis.selectivity_heatmaps("paper")
    masks={x["substrate_role"]:x for x in r.answer.tables["mask_effect_by_role"].rows}
    assert masks["nucleophile"]=={"substrate_role":"nucleophile","observed_class_pairs":67,"pairs_with_n4":0,"visible_at_n_ge4":53,"visible_at_n_ge5":53}
    rows=r.answer.tables["nucleophile_visible_by_median"].rows
    assert len(rows)==53 and [x["median_ee"] for x in rows]==sorted([x["median_ee"] for x in rows],reverse=True)
    indexed={(x["organocatalyst_class"],x["nucleophile_computed_class"]):x for x in rows}
    assert indexed[("amino_acid","ketone")]["n_reactions"]==172
    assert indexed[("proline_pyrrolidine_derivatives","ketone")]["n_reactions"]==616
    assert indexed[("proline_pyrrolidine_derivatives","ketone")]["median_ee"]==90


def test_complete_record_comparison_preserves_missing_rows_and_nulls():
    from chem_repro.common.comparison import compare_record_tables
    from chem_repro.common.contracts import Table
    a=Table(description="left",columns=["id","tier"],rows=[{"id":1,"tier":None},{"id":2,"tier":1}])
    b=Table(description="right",columns=["id","tier"],rows=[{"id":1,"tier":None},{"id":3,"tier":None}])
    rows=compare_record_tables(a,b,id_column="id",fields=["tier"]).rows
    assert len(rows)==3 and rows[0]["changed_fields"]==[]
    assert rows[1]["observed__present"] is False and rows[1]["changed_fields"]==["record_presence"]
    assert rows[2]["reference__present"] is False
    duplicate=a.model_copy(update={"rows":[a.rows[0],a.rows[0]]})
    with pytest.raises(ValueError,match="unique"):compare_record_tables(duplicate,b,id_column="id",fields=["tier"])
    preview=a.model_copy(update={"is_preview":True})
    with pytest.raises(ValueError,match="previews"):compare_record_tables(preview,b,id_column="id",fields=["tier"])


def test_priority_comparison_computes_every_identity_and_changed_tier1_record(monkeypatch,tmp_path):
    from chem_repro.heracleum.server import prioritize_candidates
    from chem_repro.heracleum.analysis import prioritize
    import csv
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR",str(tmp_path))
    result=prioritize_candidates("compare",include_figures=False)
    comparison=result.answer.tables["identity_policy_record_comparison"]
    with (tmp_path/(comparison.full_table_artifact_sha256+".csv")).open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==102 and len({r["synID"] for r in rows})==102
    changed=result.answer.tables["changed_snapshot_tier1_records"].rows
    assert len(changed)==3
    assert all(r["published_snapshot__tier"]==1 and r["strict__tier"] is None and r["changed_fields"]==["tier"] for r in changed)
    actual={m.name:m.value for m in result.answer.metrics}
    assert actual["records_with_tier_change"]==65 and actual["records_without_tier_change"]==37
    summary=result.answer.tables["identity_policy_summary"]
    assert summary.is_preview is False
    assert summary.rows == [
        {"identity_policy":"published_snapshot","tier_1":21,"tier_2":24,"tier_3":57,"unranked":0,"ranked":102,"total_records":102},
        {"identity_policy":"strict","tier_1":18,"tier_2":10,"tier_3":9,"unranked":65,"ranked":37,"total_records":102},
    ]
    original={r["synID"]:r for r in prioritize("published_snapshot").answer.tables["candidates"].rows}
    strict={r["synID"]:r for r in prioritize("strict").answer.tables["candidates"].rows}
    assert {int(r["synID"]) for r in rows if r["published_snapshot__tier"]!=r["strict__tier"]}=={key for key in original if original[key]["tier"]!=strict[key]["tier"]}


def test_candidate_rules_and_complete_shortlists_do_not_change_rankings():
    from chem_repro.heracleum.analysis import prioritize
    for mode,expected in [("published_snapshot",{1:21,2:24,3:57,None:0}),("strict",{1:18,2:10,3:9,None:65})]:
        result=prioritize(mode)
        rows=result.answer.tables["candidates"].rows
        assert len(rows)==102
        assert all(sum(r["tier"]==tier for r in rows)==count for tier,count in expected.items())
        shortlist=result.answer.tables["tier1_shortlist"].rows
        assert len(shortlist)==expected[1]
        assert {r["synID"] for r in shortlist}=={r["synID"] for r in rows if r["tier"]==1}
        assert all(r["criterion_checks"]["tier1_docking"] is True for r in rows if r["tier"]==1)
        if mode=="strict":
            assert sum(r["unranked_reasons"]==["identity_conflict"] for r in rows)==65
        rules=result.metadata.protocol_parameters["tier_rule_parameters"]
        assert rules=={"tier1_min_favorable":6,"tier1_max_strong_targets":2,"tier1_max_brenk":1,"tier2_min_favorable":5,"required_pains":0}
        assert len(result.metadata.protocol_parameters["informative_endpoints"])==8
        assert "canonical isomeric SMILES" in result.metadata.protocol_parameters["identity_verification"]
