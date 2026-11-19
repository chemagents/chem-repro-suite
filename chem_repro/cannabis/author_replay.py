"""Recompute the deposited-code result from complete, checksummed held-out scores.

This is separate from the Appendix A reconstruction and final RTE390 mixture.
The training bundle records unchanged author code, new fits and checkpoint reloads.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .data import load

METHODS = {'dmpnn':'p_dmpnn', 'catboost_rmt95_standalone':'p_catboost_rmt95',
           'svm_rmt95_standalone':'p_svm_rmt95', 'product_rmt95_veto':'p_final_product_rmt95'}


def recalculate(predictions, registry, source):
    """Validate every held-out ID, label and product before calculating metrics."""
    expected_splits = set(registry.split_id.unique())
    if set(predictions.split_id.unique()) != expected_splits:
        raise ValueError('Predictions do not cover every deposited split')
    if predictions.duplicated(['split_id','ligand_id']).any():
        raise ValueError('Duplicate split/ligand prediction')
    if source.ligand_id.duplicated().any():
        raise ValueError('Duplicate source ligand identity')
    indexed = source.set_index('ligand_id')
    rows = []
    for split_id, group in predictions.groupby('split_id', sort=True):
        original = registry.loc[registry.split_id == split_id]
        if original.ligand_id.duplicated().any() or set(original['set']) != {'train','test'}:
            raise ValueError('Malformed deposited split')
        train_ids = set(original.loc[original['set']=='train','ligand_id'])
        test_ids = set(original.loc[original['set']=='test','ligand_id'])
        if train_ids & test_ids or set(group.ligand_id) != test_ids:
            raise ValueError('Held-out IDs differ from the deposited split')
        labelled = set(source.loc[source.activity.notna(), 'ligand_id'])
        if train_ids | test_ids != labelled:
            raise ValueError('Deposited split does not cover the labeled cohort')
        y = indexed.loc[group.ligand_id,'activity'].to_numpy()
        if not np.array_equal(y, group.activity.to_numpy()):
            raise ValueError('Prediction labels differ from source labels')
        if not np.array_equal(indexed.loc[original.ligand_id,'activity'],original.activity):
            raise ValueError('Registry labels differ from source labels')
        values = group[list(METHODS.values())].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Nonfinite or out-of-range probability')
        product = group.p_dmpnn.to_numpy() * np.clip(group.p_catboost_rmt95.to_numpy(),1e-8,1)
        if not np.allclose(product,group.p_final_product_rmt95,rtol=0,atol=1e-14):
            raise ValueError('Final score is not the deposited RMT95 product')
        base_positive = group.p_dmpnn.to_numpy() >= .5
        for method, column in METHODS.items():
            p = group[column].to_numpy(float)
            positive = p >= .5
            tp=int((positive & (y==1)).sum());fp=int((positive & (y==0)).sum())
            tn=int((~positive & (y==0)).sum());fn=int((~positive & (y==1)).sum())
            rows.append({'split_id':split_id,'method':method,'n_test':len(y),'threshold':.5,
                'roc_auc':float(roc_auc_score(y,p)), 'average_precision':float(average_precision_score(y,p)),
                'fpr':fp/(fp+tn),'recall':tp/(tp+fn),'precision':tp/max(tp+fp,1),
                'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                'false_positives_removed':int((base_positive & (y==0) & ~positive).sum()),
                'true_positives_lost':int((base_positive & (y==1) & ~positive).sum())})
    per_split = pd.DataFrame(rows)
    summaries=[]
    for method,g in per_split.groupby('method',sort=False):
        row={'method':method,'n_splits':len(g)}
        for metric in ['roc_auc','average_precision','fpr','recall','precision']:
            row[metric+'_mean']=float(g[metric].mean())
            row[metric+'_sd']=float(g[metric].std(ddof=1)) if len(g)>1 else None
        for metric in ['tp','fp','tn','fn','false_positives_removed','true_positives_lost']:
            row[metric+'_total']=int(g[metric].sum())
        summaries.append(row)
    return per_split, pd.DataFrame(summaries)


def author_code_replay():
    manifest=verified_manifest('cannabis_author')
    required={'heldout_predictions.csv','structural_components.csv','training_receipt.json','checkpoint_verification.json','training_request.json'}
    if not required <= set(manifest['files']):
        raise ValueError('Complete author-code training bundle is not installed; no partial run is promoted')
    root=DATA_ROOT/'cannabis_author'
    receipt=json.loads((root/'training_receipt.json').read_text())
    verified=json.loads((root/'checkpoint_verification.json').read_text())
    request=json.loads((root/'training_request.json').read_text())
    if receipt.get('complete') is not True or receipt.get('splits')!=30:
        raise ValueError('Author-code training receipt is incomplete')
    expected={f'split_{i:02d}' for i in range(30)}
    if len(verified)!=30 or {r['split_id'] for r in verified}!=expected:
        raise ValueError('Missing independent checkpoint verification')
    if any(r.get('source_test_ids_match') is not True or r.get('test_train_overlap')!=0 for r in verified):
        raise ValueError('Checkpoint verification reports invalid split membership')
    prediction=pd.read_csv(root/'heldout_predictions.csv')
    registry=pd.read_csv(DATA_ROOT/'cannabis/split_registry.csv')
    per_split,summary=recalculate(prediction,registry,load())
    components=pd.read_csv(root/'structural_components.csv')
    merged=prediction.merge(components,on=['split_id','ligand_id','activity'],validate='one_to_one')
    if len(merged)!=len(prediction) or len(components)!=len(prediction):
        raise ValueError('Structural components do not cover every held-out row')
    weights={r['split_id']:r['blend_weight_dmpnn'] for r in verified}
    weight=merged.split_id.map(weights).to_numpy(float)
    recomposed=weight*merged.dmpnn_proba+(1-weight)*merged.hgb_proba
    if not np.allclose(recomposed,merged.p_dmpnn,atol=1e-7,rtol=1e-6):
        raise ValueError('Source score does not match its recorded DMPNN/HGB mixture')
    pure=merged[list(prediction.columns)].copy()
    pure['p_dmpnn']=merged.dmpnn_proba
    pure['p_final_product_rmt95']=pure.p_dmpnn*np.clip(pure.p_catboost_rmt95,1e-8,1)
    pure_per,pure_summary=recalculate(pure,registry,load())
    rename={'dmpnn':'pure_dmpnn','product_rmt95_veto':'pure_dmpnn_times_rmt95'}
    pure_per=pure_per.loc[pure_per.method.isin(rename)].copy()
    pure_summary=pure_summary.loc[pure_summary.method.isin(rename)].copy()
    pure_per['method']=pure_per.method.map(rename)
    pure_summary['method']=pure_summary.method.map(rename)
    blended_folds=sum(v!=1 for v in weights.values())
    refs=pd.DataFrame([
        {'source':'deposited_code_RMT95','method':'dmpnn','roc_auc':.9283,'fpr':.1220},
        {'source':'deposited_code_RMT95','method':'product_rmt95_veto','roc_auc':.9167,'fpr':.0514},
        {'source':'final_article_RMT95_RTE390_mix','method':'different_geometric_mixture','roc_auc':.9176,'fpr':.0492}])
    comparisons=[]
    for ref in refs.loc[refs.source=='deposited_code_RMT95'].itertuples():
        measured=summary.loc[summary.method==ref.method].iloc[0]
        comparisons.append({'method':ref.method,'reference_protocol':ref.source,
            'roc_auc_reference':ref.roc_auc,'roc_auc_computed':float(measured.roc_auc_mean),
            'roc_auc_difference':float(measured.roc_auc_mean-ref.roc_auc),
            'fpr_reference':ref.fpr,'fpr_computed':float(measured.fpr_mean),
            'fpr_difference':float(measured.fpr_mean-ref.fpr)})
    result=AnalysisResult(status='partial',answer=Answer(
        finding='Recomputed all30 deposited-split metrics from new fits of the unchanged author DMPNN/HGB '
                'and RMT95 veto code. All19050 held-out rows are checked against original IDs and labels; '
                'the source score named dmpnn is a validation-selected blend. Pure DMPNN metrics are separate. '
                'This repeats the deposited RMT95 branch, not the final RTE390 geometric mixture.',
        metrics=[Metric(name='source_splits_verified',value=30,population='deposited split_registry.csv'),
                 Metric(name='heldout_rows',value=len(prediction),population='30 repeated test sets; records repeat between splits'),
                 Metric(name='checkpoints_reloaded',value=30,population='new author-code fits'),
                 Metric(name='folds_with_nonzero_hgb_weight',value=blended_folds,population='30 source model-selection decisions'),
                 Metric(name='source_selection_membership_matches',value=bool(receipt['ordered_selection_matches_source']),population='95 selected residue features')],
        tables={'computed_summary':table(summary,'Mean and sample SD across30 author splits; FPR is a fraction'),
                'computed_per_split':table(per_split,'Complete confusion counts and metrics at source threshold0.5'),
                'heldout_predictions':table(prediction,'Scores from newly fitted original author code'),
                'pure_dmpnn_summary':table(pure_summary,'Pure neural scores and their RMT95 product; no refitting or threshold tuning'),
                'pure_dmpnn_per_split':table(pure_per,'Pure DMPNN results separated from the source validation-selected HGB blend'),
                'source_references':table(refs,'Separate deposited-code and final-paper references',evidence_kind='paper_reported'),
                'same_protocol_comparison':table(pd.DataFrame(comparisons),'Only RMT95 references compared with RMT95 results'),
                'checkpoint_checks':table(pd.DataFrame(verified),'Independent checkpoint reloads; source IDs and labels verified')},
        limitations=[EvidenceNote(code='DEPOSITED_CODE_VS_FINAL_PAPER',message='The deposited RMT95 product code explicitly excludes '
            'the later RTE390 geometric mixture. Its5.14% FPR reference must not be replaced by4.92%.'),
            EvidenceNote(code='SOURCE_SELECTION_USES_TEST_LABELS',message='Unchanged source selection uses all3171 labeled records, '
            'including outer-test labels; its inner diagnostic fits/evaluates logistic regression on validation rows. '
            'This is a fidelity replay, not leakage-free independent validation.'),
            EvidenceNote(code='NEW_AUTHOR_CODE_FITS',message='Weights were retrained with deposited defaults on CPU; '
            'historical30 checkpoints were not recovered. Original defaults differ from the paper supplement configuration.'),
            EvidenceNote(code='SOURCE_DMPNN_SCORE_IS_SELECTED_BLEND',message=f'{blended_folds}/30 new source fits select a nonzero '
            'HGB weight. The source veto reads blend_proba under the name dmpnn. Both that actual source behavior '
            'and pure neural predictions are reported separately; neither is selected by closeness to the article.')]),
        metadata=provenance('cannabis_author','cannabis.deposited_author_code.replay.v1',
            source_revision=receipt['source_commit'],training_environment=request,threshold=.5))
    result.metadata.sources.extend(provenance('cannabis','cannabis.source.ids').sources)
    result.metadata.reused_from_cache=True
    return result
