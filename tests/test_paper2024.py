import numpy as np
import pandas as pd
import pytest

from chem_repro.cannabis.paper2024 import (
    catboost_search_parameters,
    descriptor_frame,
    prepare,
    xgboost_search_space,
)
from chem_repro.common.fragment_selection import FragmentSelector


def records(smiles, labels, sources=None):
    frame = pd.DataFrame({"source_id": [str(i) for i in range(len(smiles))], "smiles": smiles, "y": labels})
    if sources is not None:
        frame["source_dataset"] = sources
    return frame


def test_classification_conflicts_and_chemical_exclusions_preserve_every_record():
    data = records(["CCO", "OCC", "C[N+](C)(C)C.[Cl-]", "CC[Na]", "O", "bad", "CC"],
                   [0, 1, 1, 0, 1, 0, 1])
    result = prepare(data, "classification", "source")
    assert len(result.records) == len(data) == 7
    assert result.summary["conflicting_label_records"] == 2
    assert result.summary["chemistry_rejected_records"] == 3
    assert result.summary["aggregated_molecules"] == 2
    assert set(result.records.loc[result.records.status == "label_conflict", "raw_smiles"]) == {"CCO", "OCC"}
    assert any("[Cl-]" in s for s in result.molecules.smiles)


def test_identity_retains_isotopes_stereochemistry_and_collapse_is_audited():
    result = prepare(records(["CCO", "OCC", "[13CH3]CO", "C[C@H](O)Cl", "C[C@@H](O)Cl", "[CH3:8][CH2:9]O"], [1]*6),
                     "classification", "source")
    assert len(result.molecules) == 4
    assert result.summary["duplicate_records_collapsed"] == 2
    assert int(result.records.atom_maps_removed.sum()) == 1
    assert any("13" in s for s in result.molecules.smiles)


def test_regression_source_weighting_is_not_record_weighting():
    data = records(["CCO", "OCC", "CCO"], [1, 3, 10], ["A", "A", "B"])
    simple = prepare(data, "regression", "unused", "record_mean").molecules.iloc[0]
    balanced = prepare(data, "regression", "unused", "source_balanced_mean").molecules.iloc[0]
    assert simple.y == pytest.approx(14/3)
    assert balanced.y == 6
    assert simple.source_balanced_mean == balanced.source_balanced_mean
    assert simple.n_sources == 2


def test_labels_are_not_silently_imputed_or_reinterpreted():
    with pytest.raises(ValueError, match="Targets"):
        prepare(records(["CC"], [np.nan]), "regression", "source")
    with pytest.raises(ValueError, match="Targets"):
        prepare(records(["CC"], [2]), "classification", "source")


def test_missing_record_ids_retain_row_identity_and_literal_na_is_not_missing():
    data = records(["CCO", "OCC", "CC"], [1, 1, 0])
    data["source_id"] = [None, "", "NA"]
    result = prepare(data, "classification", "source")
    assert result.summary["missing_source_id_records"] == 2
    assert list(result.records.input_row) == [0, 1, 2]
    assert result.records.iloc[2].source_id == "NA"
    assert result.summary["retained_input_records"] == 3
    assert len(result.molecules) == 2


def test_fragment_selection_fits_only_training_and_preserves_negative_correlations():
    train = pd.DataFrame({"f1": [0., 1., 0., 1.], "duplicate": [0., 2., 0., 2.],
                          "opposite": [1., 0., 1., 0.], "heldout_only": [0.]*4,
                          "descriptor": [1., 2., 3., 4.]})
    selector = FragmentSelector(["f1", "duplicate", "opposite", "heldout_only"]).fit(train)
    assert selector.selected_features_ == ["f1", "opposite", "descriptor"]
    query = train.copy()
    query["heldout_only"] = 100
    assert list(selector.transform(query)) == selector.selected_features_
    assert selector.n_training_records_ == 4
    audit = selector.selection_audit_.set_index("feature")
    assert audit.loc["duplicate", "correlated_with"] == "f1"
    with pytest.raises(ValueError, match="order"):
        selector.transform(query[query.columns[::-1]])


def test_fragment_boundaries_and_nonfinite_values():
    frame = pd.DataFrame({"at_boundary": [1.] + [0.]*39, "constant": [1.]*40, "rare": [0.]*40})
    selector = FragmentSelector(list(frame)).fit(frame)
    assert "at_boundary" in selector.selected_features_
    assert "rare" not in selector.selected_features_
    with pytest.raises(ValueError, match="Nonfinite"):
        selector.transform(frame.assign(rare=np.nan))


def test_declared_substitute_descriptors_include_counts_and_finite_values():
    frame, fragments = descriptor_frame(["CCO", "c1ccccc1", "[13CH3]C(=O)O"])
    assert len(fragments) > 50
    assert list(frame.element_C) == [2, 6, 2]
    assert frame.loc[1, "bonds_aromatic"] == 6
    assert frame.loc[0, "HeavyAtomCount"] == 3
    assert np.isfinite(frame.to_numpy()).all()


@pytest.mark.parametrize("bootstrap", ["Bayesian", "Bernoulli", "MVS"])
def test_catboost_bootstrap_parameters_are_conditionally_compatible(bootstrap):
    class Trial:
        def suggest_float(self, name, low, high, **kwargs):
            return low

        def suggest_int(self, name, low, high):
            return low

        def suggest_categorical(self, name, values):
            return bootstrap if name == "bootstrap_type" else values[0]

    p = catboost_search_parameters(Trial())
    assert ("bagging_temperature" in p) == (bootstrap == "Bayesian")
    assert ("subsample" in p) == (bootstrap != "Bayesian")


def test_search_size_exposes_full_cost_instead_of_silently_truncating_grid():
    assert len(xgboost_search_space(full=True)) == 67392
    reduced = list(xgboost_search_space(full=False))
    assert len(reduced) == 4
    assert all(p["n_estimators"] == 2000 and p["max_depth"] == 12 for p in reduced)
