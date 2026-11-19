"""Recover local author records without inferring the model behind their score."""
import json
import pandas as pd
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .data import load


def align_archive(archive, source):
    if archive.ligand_id.isna().any() or archive.ligand_id.duplicated().any():
        raise ValueError('Archive ligand IDs must be unique and present')
    joined = archive.merge(source, on='ligand_id', suffixes=('_archive', '_source'),
                           how='left', validate='one_to_one', indicator=True)
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Archive contains IDs absent from the deposited dataset')
    if not joined.activity.isna().all():
        raise ValueError('Archive membership includes labeled reference compounds')
    if not (joined.SMILES_archive == joined.SMILES_source).all():
        raise ValueError('Archive structures differ from deposited structures')
    same_id = joined.synID_archive.eq(joined.synID_source) | (
        joined.synID_archive.isna() & joined.synID_source.isna())
    if not same_id.all():
        raise ValueError('Archive synID values differ from the deposited values')
    score = pd.to_numeric(joined.precision, errors='raise')
    if score.isna().any() or not score.between(0, 1).all():
        raise ValueError('Archived score must be finite and within [0,1]')
    return joined.drop(columns='_merge')


def recovered_cohort_audit():
    verified_manifest('cannabis_author')
    archive = pd.read_csv(DATA_ROOT / 'cannabis_author/victor.csv')
    source = load()
    joined = align_archive(archive, source)
    archive_ids = set(joined.ligand_id)
    excluded = source.loc[source.activity.isna() & ~source.ligand_id.isin(archive_ids)].copy()
    membership = joined[['ligand_id', 'synID_source', 'SMILES_source', 'precision',
                         'superclass', 'class', 'class.1']].rename(columns={
        'synID_source':'synID', 'SMILES_source':'SMILES', 'precision':'archived_precision'})
    membership['archived_score_ge_070'] = membership.archived_precision >= .70
    verified_manifest('cannabis_candidates')
    figure4=json.loads((DATA_ROOT/'cannabis_candidates/identity_map.json').read_text())['records']
    by_id=membership.set_index('ligand_id')
    comparisons=[]
    for record in figure4:
        ligand_id=record['source_ligand_id']
        value=float(by_id.loc[ligand_id,'archived_precision']) if ligand_id in by_id.index else None
        reference=record['paper_final_score']
        comparisons.append({'paper_rank':record['paper_rank'],'paper_name':record['paper_name'],
            'source_ligand_id':ligand_id,'present_in_local_archive':ligand_id in by_id.index,
            'archived_precision':value,'paper_final_ensemble_score':reference,
            'archive_minus_paper':None if value is None else value-reference,
            'same_score_definition_established':False})
    metrics = [Metric(name='recovered_local_records', value=len(joined), population='local victor.csv'),
               Metric(name='exact_raw_SMILES_matches', value=len(joined), population='local archive vs deposited data.csv'),
               Metric(name='source_query_records_outside_local_archive', value=len(excluded), population='2749 source query records'),
               Metric(name='archived_score_ge_070', value=int(membership.archived_score_ge_070.sum()),
                      population='local victor.csv, archived precision field'),
               Metric(name='missing_synID_in_both_sources', value=int(joined.synID_source.isna().sum()),
                      population='local victor.csv'),
               Metric(name='figure4_candidates_below_archived_score_070',
                      value=sum(r['archived_precision'] is not None and r['archived_precision']<.70 for r in comparisons),
                      population='10 mapped Figure4 candidates; archived precision field')]
    return AnalysisResult(status='partial', answer=Answer(
        finding='Recovered an explicit 2465-record local author table with matching deposited IDs and raw SMILES. '
                'Its archived precision field yields 1010 records at >=0.70. These are recomputed archive counts; '
                'the model producing that field and equivalence to the final article DMPNN are not established.',
        metrics=metrics,
        tables={'recovered_membership':table(membership, 'Actual local membership; archived score is not a newly fitted probability', evidence_kind='source_record'),
                'excluded_source_query_records':table(excluded, '284 source query records absent from local victor.csv'),
                'figure4_archive_comparison':table(pd.DataFrame(comparisons),
                    'All10 published candidate graphs aligned to local archive; score lineage remains unresolved'),
                'published_counts':table(pd.DataFrame([{'quantity':'docked_DS1','value':2465},
                    {'quantity':'DMPNN_positive','value':1010}, {'quantity':'secondary_metabolites','value':44}]),
                    'Article references; no independent recovery of all44 secondary labels', evidence_kind='paper_reported')},
        limitations=[EvidenceNote(code='ARCHIVED_SCORE_LINEAGE_UNRESOLVED',
            message='The local file calls its score precision; neither the producing checkpoint nor exact score definition '
                    'is established. Two published top10 candidates have archived scores below0.70; treating precision '
                    'as the initial DMPNN screen would therefore not reproduce their inclusion. Numerical count agreement '
                    'alone is not proof of score provenance.'),
                     EvidenceNote(code='CLASS_ANNOTATIONS_NOT_CCD_STATUS',
            message='Local superclass/class annotations are retained as recorded. They are not silently converted '
                    'into CCD secondary-metabolite or biological occurrence annotations.')]),
        metadata=provenance('cannabis_author','cannabis.author_archive.cohort_alignment.v1',threshold=.70))
