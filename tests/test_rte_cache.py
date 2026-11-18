import pytest

from chem_repro.cannabis import cached_validation


def test_default_validation_keeps_all_thirty_source_splits(monkeypatch):
    monkeypatch.delenv('CHEM_REPRO_RTE_RESULTS_DIR',raising=False)
    calls=[]
    monkeypatch.setattr(cached_validation,'validate_rte',lambda indices,**kw:calls.append((indices,kw)) or 'computed')
    monkeypatch.setattr(cached_validation,'export_tables',lambda value,**kw:value)
    assert cached_validation.validate_rte_cached()=='computed'
    assert calls[0][0]==list(range(30))
    cached_validation.validate_rte_cached([3,8],iterations=20)
    assert calls[1][0]==[3,8] and calls[1][1]['iterations']==20


def test_invalid_source_split_requests_do_not_launch_training(monkeypatch):
    monkeypatch.setattr(cached_validation,'validate_rte',lambda *a,**kw:pytest.fail('unexpected training'))
    for indices in [[],[0,0],[-1],[30],['0']]:
        with pytest.raises(ValueError):cached_validation.validate_rte_cached(indices)


def test_explicit_new_budget_fits_without_overwriting_historical_results(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    historic=tmp_path/'split_00.json'
    body=json.dumps({'metadata':{'protocol_parameters':{'catboost':{'iterations':500}}}})
    historic.write_text(body)
    monkeypatch.setenv('CHEM_REPRO_RTE_RESULTS_DIR',str(tmp_path))
    calls=[]
    output=SimpleNamespace(answer=SimpleNamespace(limitations=[]))
    monkeypatch.setattr(cached_validation,'validate_rte',lambda indices,**kw:calls.append((indices,kw)) or output)
    monkeypatch.setattr(cached_validation,'export_tables',lambda value,**kw:value)
    assert cached_validation.validate_rte_cached([0],iterations=10) is output
    assert calls[0][1]['iterations']==10
    assert output.answer.limitations[0].code=='NEW_REQUESTED_TRAINING_BUDGET'
    assert historic.read_text()==body
