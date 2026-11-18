import json

import numpy as np
import pandas as pd
import pytest

from chem_repro.cannabis import fusion


def test_identity_aligned_product_reorders_and_rejects_missing_and_duplicate_rows():
    a=pd.DataFrame({'ligand_id':[1,2,3,4], 'activity':[0,0,1,1], 'probability':[.8,.2,.9,.6]})
    b=pd.DataFrame({'ligand_id':[4,3,2,1], 'activity':[1,1,0,0], 'probability':[.4,.9,.1,.4]})
    frame,metrics=fusion.aligned_scores(a,b,expected_ids=[1,2,3,4])
    assert np.allclose(frame.product_score,[.32,.02,.81,.24])
    assert metrics['structure']['fp']==1 and metrics['product']['fp']==0
    assert metrics['structure']['tp']==2 and metrics['product']['tp']==1
    assert metrics['operating_point_effect']['delta_recall']==-.5
    for invalid in [b.iloc[:3],pd.concat([b,b.iloc[:1]])]:
        with pytest.raises(ValueError):fusion.aligned_scores(a,invalid,expected_ids=[1,2,3,4])
    b.loc[b.ligand_id==1,'activity']=1
    with pytest.raises(ValueError,match='labels disagree'):
        fusion.aligned_scores(a,b,expected_ids=[1,2,3,4])


def test_actual_rte_fit_keeps_outer_labels_out_of_training(tmp_path,monkeypatch):
    rng=np.random.default_rng(44)
    source=pd.DataFrame({'ligand_id':range(32),'activity':[0,1]*15+[np.nan,np.nan],
                         'SMILES':['CC']*32})
    values=pd.DataFrame(rng.normal(-1,.2,(32,3)),index=range(32),columns=['A','B','C'])
    captured=[]
    original=fusion.fit_residue_selector
    def selector(x,y,names,**kwargs):
        captured.append((x.copy(),y.copy()))
        return original(x,y,names,**kwargs)
    monkeypatch.setattr(fusion,'fit_residue_selector',selector)
    monkeypatch.setattr(fusion,'residue_matrix',lambda target:values)
    run=tmp_path/'structural';fold=run/'fold_00';fold.mkdir(parents=True)
    partitions={'fold':0,'train_ids':list(range(18)),'validation_ids':list(range(18,24)),
                'test_ids':list(range(24,30))}
    fit={'split':partitions};(fold/'result.json').write_text('{}')
    def write_structural():
        pd.DataFrame({'ligand_id':range(24,30),'activity':source.loc[24:29,'activity'],
                      'probability':[.4,.8,.3,.7,.2,.9]}).to_csv(fold/'heldout_predictions.csv',index=False)
    write_structural()
    pd.DataFrame({'ligand_id':[30,31],'probability':[.5,.8],'fold':[0,0],
                  'SMILES':['CC','CC']}).to_csv(fold/'query_predictions.csv',index=False)
    first=fusion._fit_fold(source,run,fit,tmp_path/'rte1',iterations=10)
    assert first['n_rte_train']==24 and first['n_test']==6
    assert first['model_reload_max_error']<1e-12
    assert all(np.array_equal(y,source.loc[:23,'activity'].to_numpy(int)) for _,y in captured)
    original_inputs=[(x.copy(),y.copy()) for x,y in captured];captured.clear()
    source.loc[24:29,'activity']=1-source.loc[24:29,'activity']
    write_structural()
    fusion._fit_fold(source,run,fit,tmp_path/'rte2',iterations=10)
    assert all(np.array_equal(x,a) and np.array_equal(y,b) for (x,y),(a,b) in zip(captured,original_inputs))
    p1=pd.read_csv(tmp_path/'rte1/heldout_predictions.csv')
    p2=pd.read_csv(tmp_path/'rte2/heldout_predictions.csv')
    assert np.array_equal(p1.rte_probability,p2.rte_probability)
    partition=json.loads((tmp_path/'rte1/partitions.json').read_text())
    assert set(partition['rte_train_ids']).isdisjoint(partition['test_ids'])
