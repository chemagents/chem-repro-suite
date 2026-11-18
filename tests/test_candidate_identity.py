import numpy as np
import pandas as pd
import pytest

from chem_repro.cannabis.candidate_identity import (
    activity_rows,
    identity_map,
    stereo_free_key,
    toxicity_rows,
    toxicity_tables,
)


def test_connectivity_mapping_removes_stereo_but_preserves_isotopes_and_charge():
    assert stereo_free_key("N[C@H](C)C(=O)O") == stereo_free_key("N[C@@H](C)C(=O)O")
    assert stereo_free_key("C[13CH2]O") != stereo_free_key("CCO")
    assert stereo_free_key("C[NH3+]") != stereo_free_key("CN")


def test_all_figure4_graphs_match_frozen_source_and_pubchem_snapshots():
    rows = identity_map()
    assert [r["source_ligand_id"] for r in rows] == [3928, 3799, 3633, 3687, 3786, 3480, 3632, 3826, 3555, 3766]
    assert rows[0]["labeled_reference_match_ids"] == [1154]
    assert rows[8]["labeled_reference_match_ids"] == [3230]
    assert len(rows[8]["pubchem_cids"]) == 4  # no silent choice of an alpha-bisabolol stereoisomer
    assert all(not r["CCD_annotation_independently_verified"] and not r["natural_occurrence_independently_verified"] for r in rows)


def sample():
    rows = [{"paper_rank": i, "paper_name": f"name{i}", "source_ligand_id": i,
             "source_smiles": "C" * i + "O", "toxicity_query_id": f"{i:05d}", "labeled_reference_match_ids": []}
            for i in range(1, 11)]
    scores = pd.DataFrame({"ligand_id": range(1, 11), "SMILES": [r["source_smiles"] for r in rows],
                           "structure_probability": np.linspace(.9, .45, 10), "rte_probability": np.linspace(.2, .95, 10)})
    scores["product_score"] = scores.structure_probability * scores.rte_probability
    scores["structural_rank"] = scores.structure_probability.rank(method="min", ascending=False).astype(int)
    scores["product_rank_before_annotations"] = scores.product_score.rank(method="min", ascending=False).astype(int)
    scores["rank_change_from_structure"] = scores.structural_rank - scores.product_rank_before_annotations
    scores["structure_screen_positive"] = scores.structure_probability >= .70
    return rows, scores


@pytest.mark.parametrize("failure", ["missing", "duplicate", "smiles", "product", "rank_change"])
def test_activity_identity_or_component_mismatch_cannot_be_named(failure):
    rows, scores = sample()
    assert len(activity_rows(rows, scores)) == 10
    if failure == "missing":
        scores = scores.iloc[1:]
    elif failure == "duplicate":
        scores.loc[1, "ligand_id"] = 1
    elif failure == "smiles":
        scores.loc[0, "SMILES"] = "CCN"
    elif failure == "product":
        scores.loc[0, "product_score"] = .9
    else:
        scores.loc[0, "rank_change_from_structure"] = 999
    with pytest.raises(ValueError):
        activity_rows(rows, scores)


def test_named_candidate_ranks_join_by_source_id_not_table_position():
    rows, scores = sample()
    shuffled = scores.sample(frac=1, random_state=7)
    result = activity_rows(rows, shuffled)
    expected = scores.set_index("ligand_id")
    for row in result.itertuples():
        original = expected.loc[row.source_ligand_id]
        assert row.source_structural_rank == original.structural_rank
        assert row.source_product_rank_before_annotations == original.product_rank_before_annotations
        assert row.source_rank_change_from_structure == original.rank_change_from_structure
        assert row.source_structure_screen_positive == original.structure_screen_positive


def test_rank_direction_describes_source_position_not_paper_position():
    rows, scores = sample()
    result = activity_rows(rows, scores)
    # Multiplication demotes the highest structural score and promotes the lowest.
    first, last = result.iloc[0], result.iloc[-1]
    assert first.source_product_rank_before_annotations > first.source_structural_rank
    assert first.source_rank_change_direction == "worsened"
    assert last.source_product_rank_before_annotations < last.source_structural_rank
    assert last.source_rank_change_direction == "improved"
    assert result.source_rank_population_size.eq(10).all()
    assert result.source_rank_comparison.str.contains("not_vs_paper_rank").all()


def test_candidate_endpoint_matrix_retains_all_values_and_unknown_flags_are_refused():
    rows, _ = sample()
    predictions = pd.DataFrame({"query_id": [r["toxicity_query_id"] for r in rows], "smiles": [r["source_smiles"] for r in rows],
                                "prediction": np.linspace(.1, 1, 10), "reference_domain_score_percent": [50.] * 10,
                                "training_identity_overlap": [False] * 10, "training_scaffold_overlap": [True] * 10,
                                "outside_reference_distance": [False] * 10})
    records = []
    for endpoint in ["a", "b"]:
        records += toxicity_rows(rows, predictions, endpoint, {"task": "classification", "unit": "probability"})
    full, matrices = toxicity_tables(records)
    assert len(full) == 20 and len(matrices["figure4_prediction_matrix"].rows) == 10
    assert matrices["figure4_prediction_matrix"].rows[9]["a"] == 1
    assert matrices["figure4_training_identity_overlap"].rows[9]["b"] is False
    with pytest.raises(ValueError, match="exactly once"):
        toxicity_tables(records[:-1])
    predictions["training_identity_overlap"] = "False"
    with pytest.raises(ValueError, match="explicit booleans"):
        toxicity_rows(rows, predictions, "a", {"task": "classification", "unit": "probability"})
