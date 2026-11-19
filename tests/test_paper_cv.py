import json
import numpy as np
import pandas as pd
import pytest
from chem_repro.cannabis import paper_cv


def cohort():
    return pd.DataFrame({'ligand_id': np.arange(60), 'activity': [0, 1] * 30})


def test_tenfold_coverage_is_not_repeated_holdout():
    frame = cohort()
    pairs = paper_cv.partitions(frame)
    assert len(pairs) == 10
    test_ids = []
    for train, test in pairs:
        assert not set(train.ligand_id) & set(test.ligand_id)
        assert len(train) + len(test) == len(frame)
        test_ids.extend(test.ligand_id)
    assert sorted(test_ids) == frame.ligand_id.tolist()
    assert [te.ligand_id.tolist() for _, te in pairs] == [te.ligand_id.tolist() for _, te in paper_cv.partitions(frame)]


def test_fold_predictions_do_not_use_test_labels():
    frame = cohort()
    train, test = paper_cv.partitions(frame)[0]
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame(rng.normal(size=(60, 3)), index=frame.ligand_id,
                          columns=['ALA1', 'LEU2', 'GLY3'])
    a = paper_cv.fit_fold(train, test, index=0, iterations=10, matrices={'TEST': matrix})
    flipped = test.copy(); flipped.activity = 1 - flipped.activity
    b = paper_cv.fit_fold(train, flipped, index=0, iterations=10, matrices={'TEST': matrix})
    assert a['selections'] == b['selections']
    assert [p['score'] for p in a['predictions']] == [p['score'] for p in b['predictions']]
    assert [p['label'] for p in a['predictions']] != [p['label'] for p in b['predictions']]
    with pytest.raises(ValueError, match='overlap'):
        paper_cv.fit_fold(train, train, index=0, iterations=10, matrices={'TEST': matrix})


def test_full_cv_cache_oof_and_artifacts(tmp_path, monkeypatch):
    frame = cohort()
    rng = np.random.default_rng(4)
    matrix = pd.DataFrame(rng.normal(size=(60, 3)), index=frame.ligand_id,
                         columns=['ALA1', 'LEU2', 'GLY3'])
    monkeypatch.setattr(paper_cv, 'load', lambda: frame)
    monkeypatch.setattr(paper_cv, 'TARGETS', {'TEST': None})
    monkeypatch.setattr(paper_cv, 'residue_matrix', lambda target: matrix)
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR', str(tmp_path/'artifacts'))
    monkeypatch.setenv('CHEM_REPRO_PAPER_RTE_RESULTS_DIR', str(tmp_path/'cache'))
    output = paper_cv.validate_paper_cv(iterations=10)
    assert not output.metadata.reused_from_cache
    rows = output.answer.tables['pooled_oof'].rows
    assert len(rows) == 4 and all(r['n_records'] == 60 for r in rows)
    assert len(output.answer.tables['fold_metrics'].rows) == 40
    assert len(output.artifacts) == 5
    monkeypatch.setattr(paper_cv, 'fit_fold', lambda *a, **kw: pytest.fail('cache replay trained'))
    replay = paper_cv.validate_paper_cv(iterations=10)
    assert replay.metadata.reused_from_cache
    assert replay.answer.tables['fold_metrics'].rows == output.answer.tables['fold_metrics'].rows
    entry = next((tmp_path/'cache').rglob('fold_00.json'))
    content = json.loads(entry.read_text());content['result']['predictions'][0]['score'] += .1
    entry.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="provenance"):paper_cv.validate_paper_cv(iterations=10)


def test_invalid_paper_fold_rejected_before_fit(monkeypatch):
    monkeypatch.setattr(paper_cv, 'load', lambda: pytest.fail('should reject before loading'))
    for indices in [[10], [0, 0], [], [-1], [True]]:
        with pytest.raises(ValueError):paper_cv.validate_paper_cv(indices)
