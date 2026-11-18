import pytest

from chem_repro.common.comparison import compare_identifier_sets
from chem_repro.heracleum.analysis import prioritize
from chem_repro.heracleum.priority_reference import attach_priority_reference


def test_reference_set_comparison_does_not_hide_duplicates_or_empty_evidence():
    result = compare_identifier_sets(["A", "B"], ["B", "C"], identity_definition="fixture record IDs")
    assert (result["intersection_count"], result["union_count"], result["jaccard"]) == (1, 3, 1 / 3)
    assert result["reference_only"] == ["A"] and result["observed_only"] == ["C"]
    assert compare_identifier_sets([], [], identity_definition="fixture IDs")["jaccard"] is None
    for reference, observed, definition in [(["A", "A"], ["A"], "IDs"), (["A"], [""], "IDs"), (["A"], ["A"], "")]:
        with pytest.raises(ValueError):
            compare_identifier_sets(reference, observed, identity_definition=definition)


@pytest.mark.parametrize("policy,source_jaccard,master_jaccard", [
    ("published_snapshot", 1.0, 0.75), ("strict", 18 / 21, 18 / 21),
])
def test_paper_priority_agreement_uses_declared_structure_identity(policy, source_jaccard, master_jaccard, tmp_path, monkeypatch):
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR", str(tmp_path))
    output = prioritize(policy)
    agreements = {r["scope"]: r for r in output.answer.tables["tier1_reference_agreement"].rows}
    assert agreements["record_associated_DS2_structures"]["jaccard"] == pytest.approx(source_jaccard)
    assert agreements["prediction_master_structures"]["jaccard"] == pytest.approx(master_jaccard)
    membership = output.answer.tables["tier1_reference_membership"].rows
    assert len(membership) == 21
    assert sum(r["identity_verified"] is False for r in membership) == 3
    if policy == "strict":
        assert all(r["computed_tier"] is None for r in membership if not r["identity_verified"])
    claim = output.answer.tables["paper_reported_priority_claim"]
    assert claim.evidence_kind == "paper_reported" and claim.rows[0]["n_favorable"] == 8
    check = output.answer.tables["priority_claim_check"].rows[0]
    assert check["synID"] == 53043814 and check["identity_verified"]
    assert check["observed_n_favorable"] == 7 and not check["matches_published_claim"]
    assert check["unfavorable_endpoints"] == ["hematotoxicity"]
    assert check["probabilities"]["hematotoxicity"] > check["DS2_median_thresholds"]["hematotoxicity"]
    all_eight = output.answer.tables["all_informative_endpoints_favorable"].rows
    assert len(all_eight) == 1 and all_eight[0]["synID"] == 17537199
    assert all_eight[0]["table_s4_names"] == ["Pabulenone"]
    assert all_eight[0]["ds2_source_name"] == "Isooxypeucedanin"
    sources = {s.name: s for s in output.metadata.sources}
    assert sources["table_s4_tier1_keys.csv"].parent_sha256 == sources["supplement.pdf"].sha256
    assert sources["priority_reference.json"].parent_sha256 == sources["paper.pdf"].sha256
    source_count = len(output.metadata.sources)
    assert attach_priority_reference(output) is output and len(output.metadata.sources) == source_count
