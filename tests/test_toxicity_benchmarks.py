import pandas as pd
import pytest
from chem_repro.cannabis import toxicity_benchmarks as b
from chem_repro.cannabis.analysis import result,metric
from chem_repro.common.datasets import table


def test_complete_matrix_preserves_both_aquatic_cohorts_and_missing_reference():
    cases=b.benchmark_requests()
    assert len(cases)==11 and len(set(cases))==11
    assert {e for e,s,c in cases}==set(b.ENDPOINTS)
    assert {c for e,s,c in cases if e=='daphnia'}=={'toxric','toxric_ecotox'}
    assert [(s,c) for e,s,c in cases if e=='reproductive']==[('random','toxric')]
    with pytest.raises(ValueError):b.benchmark_requests('reproductive','source_scaffold','toxric')
    with pytest.raises(ValueError):b.benchmark_requests('ames','random','toxric_ecotox')


def test_composition_never_mixes_population_or_picks_best_model(monkeypatch):
    calls=[]
    def evaluate(e,s,c):
        calls.append((e,s,c));n=20 if c=='toxric' else 40
        value=.6 if s=='random' else .4
        return result('verified', [metric('n_records',n)], {'computed_model_metrics':table(pd.DataFrame([
            {'model':name,'metric':'roc_auc','value':value+delta,'matches_published_rounding':False}
            for name,delta in [('base',0),('fixed_mean_blend',.1),('source_meta_stack',.2)]]), 'computed')},'test')
    monkeypatch.setattr(b,'validate_open_toxicity',evaluate)
    output=b.evaluate_benchmarks()
    assert calls==b.benchmark_requests()
    assert len(output.answer.tables['computed_benchmark_summary'].rows)==33
    comparisons=output.answer.tables['same_cohort_split_comparisons'].rows
    assert len(comparisons)==15
    assert all(abs(r['scaffold_minus_random']+.2)<1e-12 for r in comparisons)
    assert all(r['n_records']==(20 if r['cohort']=='toxric' else 40) for r in comparisons)
    assert output.answer.tables['paper_references'].evidence_kind=='paper_reported'
    assert len(output.answer.tables['paper_references'].rows)==7
