import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from chem_repro.runtime import nested_toxicity_verify as verifier
from chem_repro.runtime import syntelly_nested_worker as worker


def test_nested_groups_exclude_outer_test_from_all_inner_steps():
    groups = np.repeat(np.arange(30), 2)
    y = np.tile([0, 1], 30)
    folds = worker.nested_partitions(y, groups, "classification", 5, 3, 42)
    held_out = []
    for train, test, inner in folds:
        held_out.extend(test)
        for a, b in inner:
            assert set(a) | set(b) == set(train)
            assert not set(test) & (set(a) | set(b))
            assert not set(groups[a]) & set(groups[b])
    assert sorted(held_out) == list(range(60))


def test_impossible_grouping_is_not_replaced_by_random_split():
    with pytest.raises(ValueError, match="Insufficient groups"):
        worker.partitions(np.tile([0, 1], 10), np.zeros(20), "classification", 5, 42)


def test_nested_selection_cannot_see_heldout_only_fragment():
    raw = pd.DataFrame({"common": [0, 1, 0, 1, 0, 1], "heldout": [0, 0, 0, 0, 5, 5]})
    selector, train, test = worker.fitted_features(raw, list(raw), np.arange(4), np.arange(4, 6))
    assert list(train) == list(test) == ["common"]
    assert selector.n_training_records_ == 4


def test_complete_worker_flow_roundtrips_and_records_every_oof_row(tmp_path, monkeypatch):
    # Software/control-flow test only: a dummy estimator deliberately replaces
    # the scientific tuning/models. No dummy result is a scientific validation.
    smiles = ["C1" + "C" * n + "1" for n in range(2, 32)]
    train_path, query_path = tmp_path / "train.csv", tmp_path / "query.csv"
    pd.DataFrame({"source_id": range(30), "smiles": smiles, "y": np.arange(30) % 2}).to_csv(train_path, index=False)
    pd.DataFrame({"query_id": range(30), "smiles": smiles}).to_csv(query_path, index=False)
    config = {"endpoint": "reproductive", "task": "classification", "unit": "probability",
              "training_path": str(train_path), "query_path": str(query_path),
              "training_sha256": worker.digest(train_path), "query_sha256": worker.digest(query_path),
              "output": str(tmp_path / "results"), "software": {name: "test" for name in worker.REQUIRED_SOFTWARE},
              "outer_folds": 5, "inner_folds": 3, "seed": 42, "threads": 1,
              "optuna_trials": 8, "optuna_startup_trials": 4, "max_xgboost_candidates": 4,
              "schemes": ["identity_random", "scaffold"], "regression_aggregation": "record_mean"}
    monkeypatch.setattr(worker, "version", lambda name: "test")
    monkeypatch.setattr(worker, "tune", lambda *args: {})
    monkeypatch.setattr(worker, "model", lambda *args: DummyClassifier(strategy="prior"))
    result = worker.run(config)
    assert result["model_roundtrips"] == 20
    assert result["original_author_weights"] is False
    with np.load(tmp_path / "results/oof_and_query_predictions.npz") as arrays:
        for scheme in config["schemes"]:
            assert len(arrays[scheme + "__fixed_mean_blend"]) == 30
            assert np.isfinite(arrays[scheme + "__fixed_mean_blend"]).all()
    for path, expected in result["artifacts"].items():
        assert worker.digest(tmp_path / "results" / path) == expected
    saved = json.loads((tmp_path / "results/result.json").read_text())
    assert saved["validation"] == result["validation"]
    monkeypatch.setattr(verifier, "version", lambda name: "test")
    checked = verifier.verify(tmp_path / "results", tmp_path)
    assert len(checked["model_replays"]) == 20
    assert len(checked["fold_audit"]) == 10
    assert checked["fits_performed"] == 0
    artifact = next((tmp_path / "results").rglob("*.joblib"))
    artifact.write_bytes(b"corrupt model must be refused before deserialization")
    with pytest.raises(ValueError, match="artifact changed"):
        verifier.verify(tmp_path / "results", tmp_path)
