"""Batch the open supplement's endpoint/split/cohort matrix without selecting by score."""
import pandas as pd
from ..common.comparison import compare_variants
from ..common.datasets import table
from .open_toxicity import REFERENCES, validate_open_toxicity

ENDPOINTS = ('ames', 'reproductive', 'daphnia', 'fathead')


def benchmark_requests(endpoint='all', scheme='reported', cohort='both'):
    if endpoint not in {'all', *ENDPOINTS} or scheme not in {'reported', 'random', 'source_scaffold'} or cohort not in {'both', 'toxric', 'toxric_ecotox'}:
        raise ValueError('Unknown endpoint, split scheme or cohort')
    endpoints = ENDPOINTS if endpoint == 'all' else (endpoint,)
    requests=[]
    for ep in endpoints:
        if cohort == 'toxric_ecotox' and ep not in {'daphnia','fathead'}:
            raise ValueError('Augmented cohort is available only for aquatic endpoints')
        schemes=('random','source_scaffold') if scheme == 'reported' else (scheme,)
        for sc in schemes:
            if (ep,sc) not in REFERENCES:
                if scheme == 'reported':continue
                raise ValueError('This endpoint has no reported reference for the requested split')
            cohorts=('toxric','toxric_ecotox') if cohort == 'both' and ep in {'daphnia','fathead'} else (('toxric',) if cohort == 'both' else (cohort,))
            requests.extend((ep,sc,c) for c in cohorts)
    return requests


def evaluate_benchmarks(endpoint='all', scheme='reported', cohort='both'):
    requests=benchmark_requests(endpoint,scheme,cohort)
    if len(requests)==1:
        return validate_open_toxicity(*requests[0])
    variants={};scores=[];coverage=[];references={}
    for ep,sc,c in requests:
        label=f'{ep}_{sc}_{c}'
        r=validate_open_toxicity(ep,sc,c);variants[label]=r
        n=next(m.value for m in r.answer.metrics if m.name=='n_records')
        rows=r.answer.tables['computed_model_metrics']
        if rows.is_preview:raise ValueError('Benchmark aggregation requires complete computed metrics')
        scores.extend({'endpoint':ep,'scheme':sc,'cohort':c,'n_records':n,**row} for row in rows.rows)
        references[(ep,sc)]={'endpoint':ep,'scheme':sc,'value':REFERENCES[(ep,sc)]}
        coverage.append({'endpoint':ep,'scheme':sc,'cohort':c,'n_records':n,'status':r.status,
                         'reused_from_cache':r.metadata.reused_from_cache})
    result=compare_variants(variants, protocol_id='cannabis.open_tox.supplement_matrix.v1',
        finding=f'Executed {len(requests)} declared endpoint/split/cohort replays covering '
                f'{len(references)} published Supplement S3 references. Every base model and aggregate is retained. '
                'Both aquatic cohorts are explicit; a different cohort or estimator is not treated as a pure scaffold-transfer effect.')
    result.answer.tables['computed_benchmark_summary']=table(pd.DataFrame(scores),
        'Complete actual metrics; publication-rounding flags do not select or refit a model')
    result.answer.tables['paper_references']=table(pd.DataFrame(references.values()),
        'Supplement S3 values, separate from computed metrics', evidence_kind='paper_reported')
    result.answer.tables['requested_matrix_coverage']=table(pd.DataFrame(coverage),
        'Executed endpoint/split/cohort combinations; reproductive scaffold reference is not invented')
    comparisons=[]
    frame=pd.DataFrame(scores)
    for (ep,c,model,metric),g in frame.groupby(['endpoint','cohort','model','metric']):
        a=g.loc[g.scheme=='random'];b=g.loc[g.scheme=='source_scaffold']
        if len(a)==1 and len(b)==1:
            if a.iloc[0].n_records != b.iloc[0].n_records:raise ValueError('Cohort size changed within a named source cohort')
            comparisons.append({'endpoint':ep,'cohort':c,'n_records':int(a.iloc[0].n_records),
                'model':model,'metric':metric,'random_value':float(a.iloc[0].value),
                'scaffold_value':float(b.iloc[0].value),'scaffold_minus_random':float(b.iloc[0].value-a.iloc[0].value),
                'meta_validation_independent':False if model=='source_meta_stack' else None})
    result.answer.tables['same_cohort_split_comparisons']=table(pd.DataFrame(comparisons),
        'Same endpoint/cohort/estimator comparison; source meta-stack still has the disclosed nonnested validation limitation')
    result.metadata.protocol_parameters['requested_matrix']=[{'endpoint':e,'scheme':s,'cohort':c} for e,s,c in requests]
    return result
