import pandas as pd
import pytest

from chem_repro.common.contracts import AnalysisResult, Answer, EvidenceNote, Metric, Provenance
from chem_repro.common.datasets import table
from chem_repro.heracleum import transfer_grid


def fake_fit(endpoint, model, policy, bootstrap, include_oof):
    tables={
        'validation_metrics':table(pd.DataFrame([{'scheme':s,'r2':v,'rmse':.2,'mae':.1,'r2_ci_low':v-.1,'r2_ci_high':v+.1} for s,v in [('random',.6),('scaffold',.4)]]),'metrics'),
        'fold_audit':table(pd.DataFrame([{'scheme':s,'fold':f,'generic_scaffold_overlap':0} for s in ['random','scaffold'] for f in range(5)]),'folds'),
    }
    if include_oof:
        tables['oof_predictions']=table(pd.DataFrame([{'scheme':s,'observation_id':str(i),'fold':i%5,'observed':.3,'predicted':.5} for s in ['random','scaffold'] for i in range(10)]),'predictions')
    return AnalysisResult(status='success',answer=Answer(finding='test',metrics=[Metric(name='n_records',value=10,population='fixture'),Metric(name='delta_r2',value=.2,population='fixture')],tables=tables),metadata=Provenance(study_id='heracleum_2026',paper_doi='test',protocol_id='unit.fit',protocol_parameters={'model':model,'endpoint':endpoint,'identity_policy':policy,'bootstrap':bootstrap},sources=[],software={}))


def test_default_grid_covers_all_four_combinations_and_both_schemes(monkeypatch):
    calls=[]
    def run(*args):
        calls.append(args)
        return fake_fit(*args)
    monkeypatch.setattr(transfer_grid,'transfer',run)
    output=transfer_grid.compare_transfer(identity_policy='published_snapshot',include_figures=False)
    expected={(m,e) for m in ['RF','ElasticNet'] for e in ['neurotoxicity','eye_corrosion']}
    assert {(args[1],args[0]) for args in calls}==expected
    assert all(args[2:]==('published_snapshot',1000,True) for args in calls)
    rows=output.answer.tables['validation_metrics'].rows
    assert {(r['model'],r['endpoint'],r['scheme']) for r in rows}=={(m,e,s) for m,e in expected for s in ['random','scaffold']}
    assert len(output.answer.tables['fold_audit'].rows)==40
    assert len(output.answer.tables['oof_predictions'].rows)==80
    assert len(output.answer.tables['transfer_penalties'].rows)==4
    assert {m.name:m.value for m in output.answer.metrics}=={'model_endpoint_combinations':4,'split_evaluations':8,'n_records_per_combination':10}


def test_single_combination_remains_compatible_and_invalid_grid_is_rejected(monkeypatch):
    monkeypatch.setattr(transfer_grid,'transfer',fake_fit)
    output=transfer_grid.compare_transfer('neurotoxicity','RF',include_figures=False,include_oof=False)
    assert output.metadata.protocol_id=='unit.fit'
    assert 'oof_predictions' not in output.answer.tables
    for endpoint,model in [('unknown','RF'),('both','unknown')]:
        with pytest.raises(ValueError):transfer_grid.compare_transfer(endpoint,model)


def test_grid_preserves_qualifications_from_each_child_without_repeating_shared_notes(monkeypatch):
    def run(endpoint, model, *args):
        output = fake_fit(endpoint, model, *args)
        output.answer.limitations = [
            EvidenceNote(code="SHARED", message="A qualification common to all child fits."),
            EvidenceNote(code="CHILD_SCOPE", message=f"Applies to {model}/{endpoint}."),
        ]
        return output
    monkeypatch.setattr(transfer_grid, "transfer", run)
    output = transfer_grid.compare_transfer(include_figures=False)
    notes = output.answer.limitations
    assert sum(note.code == "SHARED" for note in notes) == 1
    assert {note.message for note in notes if note.code == "CHILD_SCOPE"} == {
        f"Applies to {model}/{endpoint}."
        for model in ["RF", "ElasticNet"] for endpoint in ["neurotoxicity", "eye_corrosion"]
    }
    assert output.metadata.protocol_parameters["target_columns"] == {
        "neurotoxicity": "admetlab_tox_neurotoxicity_di", "eye_corrosion": "admetlab_tox_ec"
    }
