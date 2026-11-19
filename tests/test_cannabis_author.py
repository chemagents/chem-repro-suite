import numpy as np
import pandas as pd
import pytest

from chem_repro.cannabis.author_archive import align_archive, recovered_cohort_audit
from chem_repro.cannabis.author_replay import recalculate


def example():
    source = pd.DataFrame({'ligand_id':[1,2,3,4], 'activity':[0,1,0,1]})
    registry = pd.DataFrame({'split_id':['split_00']*4,'ligand_id':[1,2,3,4],
                             'set':['test','test','train','train'],'activity':[0,1,0,1]})
    pred = pd.DataFrame({'split_id':['split_00']*2,'ligand_id':[1,2],'activity':[0,1],
                         'p_dmpnn':[.6,.8],'p_catboost_rmt95':[.4,.4],
                         'p_svm_rmt95':[.7,.9],'p_final_product_rmt95':[.24,.32]})
    return pred,registry,source


def test_veto_reduction_does_not_hide_true_positive_loss():
    per,summary=recalculate(*example())
    product=per.set_index('method').loc['product_rmt95_veto']
    assert product.fpr==0
    assert product.recall==0
    assert product.false_positives_removed==1
    assert product.true_positives_lost==1
    assert product.roc_auc==1
    assert summary.roc_auc_sd.isna().all()


@pytest.mark.parametrize('corruption',['labels','membership','product','missing_split','nonfinite','duplicate'])
def test_replay_rejects_corrupt_evidence(corruption):
    pred,registry,source=example()
    if corruption=='labels':pred.loc[0,'activity']=1
    elif corruption=='membership':pred.loc[0,'ligand_id']=3
    elif corruption=='product':pred.loc[0,'p_final_product_rmt95']=.42
    elif corruption=='missing_split':pred['split_id']='split_01'
    elif corruption=='nonfinite':pred.loc[0,'p_dmpnn']=np.nan
    else:pred=pd.concat([pred,pred.iloc[[0]]])
    with pytest.raises(ValueError):recalculate(pred,registry,source)


def test_archive_accepts_missing_ids_but_not_wrong_structure():
    source=pd.DataFrame({'ligand_id':[1], 'synID':[np.nan], 'SMILES':['CC'], 'activity':[np.nan]})
    archive=pd.DataFrame({'ligand_id':[1], 'synID':[np.nan], 'SMILES':['CC'], 'precision':[.7]})
    assert len(align_archive(archive,source))==1
    archive.loc[0,'SMILES']='CCC'
    with pytest.raises(ValueError,match='structures'):align_archive(archive,source)


def test_real_archive_counts_are_not_promoted_to_new_model_predictions():
    r=recovered_cohort_audit()
    metrics={m.name:m.value for m in r.answer.metrics}
    assert metrics['recovered_local_records']==2465
    assert metrics['archived_score_ge_070']==1010
    assert metrics['source_query_records_outside_local_archive']==284
    assert r.answer.tables['recovered_membership'].evidence_kind=='source_record'
    assert 'ARCHIVED_SCORE_LINEAGE_UNRESOLVED' in {n.code for n in r.answer.limitations}
