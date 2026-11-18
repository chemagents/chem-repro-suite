import json
import shutil
from pathlib import Path

import pytest

from chem_repro.cannabis.open_toxicity import validate_open_toxicity


@pytest.fixture
def replay(tmp_path, monkeypatch):
    source = Path(__file__).parent / "fixtures/open_tox/reproductive_random"
    target = tmp_path / "runs/reproductive_random"
    shutil.copytree(source, target)
    monkeypatch.setenv("CHEM_REPRO_OPEN_TOX_RESULTS_DIR", str(target.parent))
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    return target


def test_real_cached_replay_distinguishes_reference_and_meta_estimator(replay):
    result = validate_open_toxicity("reproductive", "random")
    rows = {r["model"]: r for r in result.answer.tables["computed_model_metrics"].rows}
    assert rows["cat_ecfp"]["value"] == pytest.approx(0.58615136876)
    assert rows["cat_ecfp"]["matches_published_rounding"]
    assert not rows["source_meta_stack"]["matches_published_rounding"]
    assert rows["source_meta_stack"]["value"] == pytest.approx(0.426328502415)
    assert result.metadata.reused_from_cache
    assert result.metadata.study_id == "cannabis_open_tox_2026"
    assert result.metadata.software["scikit-learn"] == "1.9.0"
    assert (
        sum(
            r["meta_test_labels_used_in_base_training_for_meta_train_features"]
            for r in result.answer.tables["meta_validation_audit"].rows
        )
        == 156
    )


def test_modified_metric_cannot_pass_as_an_observed_result(replay):
    file = replay / "result.json"
    result = json.loads(file.read_text())
    result["metrics"]["source_meta_stack"] = 0.99
    file.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="disagrees"):
        validate_open_toxicity("reproductive", "random")


def test_wrong_source_fingerprint_is_rejected(replay):
    file = replay / "result.json"
    result = json.loads(file.read_text())
    result["run_spec"]["source_sha256"] = "0" * 64
    file.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="provenance"):
        validate_open_toxicity("reproductive", "random")
