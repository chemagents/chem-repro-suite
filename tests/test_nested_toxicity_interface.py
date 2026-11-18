import hashlib
import io
import json
import zipfile

import pytest

from chem_repro.cannabis import nested_toxicity, server


def test_nested_mode_refuses_missing_requested_endpoints_instead_of_falling_back(tmp_path, monkeypatch):
    registry = tmp_path / "runs.json"
    registry.write_text(json.dumps({"reproductive": {"result_dir": "unread", "source_dir": "unread"}}))
    monkeypatch.setenv("CHEM_REPRO_NESTED_TOX_RUNS_FILE", str(registry))
    with pytest.raises(RuntimeError, match="ames"):
        nested_toxicity.evaluate(["reproductive", "ames"])


def test_existing_tool_routes_explicit_nested_recipe(monkeypatch):
    received = []
    marker = object()

    def evaluate(endpoints):
        received.append(endpoints)
        return marker

    monkeypatch.setattr(nested_toxicity, "evaluate", evaluate)
    assert server.open_toxicity_profile(["reproductive"], "nested_paper2024_v2") is marker
    assert received == [["reproductive"]]
    with pytest.raises(ValueError, match="Unknown"):
        server.open_toxicity_profile(["reproductive"], "invented_recipe")


def test_search_audit_keeps_failed_trials_distinct_from_successful_inner_fits(tmp_path):
    directory = tmp_path / "scaffold_fold0"
    directory.mkdir()
    values = {
        "optuna_trials.json": [
            {"number": 0, "state": "COMPLETE"},
            {"number": 1, "state": "FAIL", "parameters": {"subsample": .1},
             "user_attrs": {"backend_failure": "Too few sampling units"}}],
        "CatBoost_ECFP_completed_candidates.json": [{"number": 0, "inner_fold_metrics": [{}, {}, {}]}],
        "XGBoost_fragments_completed_candidates.json": [{"number": i, "inner_fold_metrics": [{}, {}, {}]} for i in range(4)],
    }
    result = {"config": {"endpoint": "reproductive", "schemes": ["scaffold"], "outer_folds": 1, "optuna_trials": 2},
              "artifacts": {}}
    for name, value in values.items():
        path = directory / name
        path.write_text(json.dumps(value))
        result["artifacts"]["scaffold_fold0/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, failures = nested_toxicity.search_audit(tmp_path, result)
    assert rows[0]["requested_candidates"] == 2
    assert rows[0]["completed_candidates"] == 1
    assert rows[0]["failed_candidates"] == 1
    assert rows[0]["successful_candidate_inner_fold_evaluations"] == 3
    assert rows[1]["exhaustive_finite_grid"]
    assert failures[0]["backend_error"] == "Too few sampling units"


def test_model_bundle_hash_does_not_change_with_wall_clock(monkeypatch):
    contents = []
    for year in [2025, 2026]:
        monkeypatch.setattr(zipfile.time, "localtime", lambda *args, y=year: (y, 1, 1, 0, 0, 0, 0, 1, 0))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            nested_toxicity.zip_member(archive, "model.bin", b"immutable fitted bytes")
        contents.append(buffer.getvalue())
    assert contents[0] == contents[1]
