import numpy as np
import pytest
from chem_repro.heracleum.cv_registry import reference_fold_assignments
from chem_repro.heracleum.data import load
from chem_repro.common.chemistry import scaffold

@pytest.mark.parametrize('policy',['published_snapshot','strict'])
def test_reference_splits_preserve_exact_cohort_and_scaffold_separation(policy):
    frame,_=load(policy);ids=frame.synID.astype(str).tolist()
    groups=(frame.generic_murcko_scaffold.fillna('none').to_numpy() if policy=='published_snapshot' else frame.canonical_smiles.map(lambda s:scaffold(s,generic=True,retain_stereo=True)).to_numpy())
    registry=reference_fold_assignments(ids,groups,policy)
    for scheme,folds in registry.items():
        assert len(folds)==len(ids) and set(folds)==set(range(5))
        for fold in range(5):
            test=folds==fold
            if scheme=='scaffold':assert not set(groups[test])&set(groups[~test])
    with pytest.raises(ValueError,match='cohort|membership'):
        reference_fold_assignments(ids[::-1],groups[::-1],policy)
    wrong=groups.copy();wrong[0]='different-group'
    with pytest.raises(ValueError,match='membership'):
        reference_fold_assignments(ids,wrong,policy)
