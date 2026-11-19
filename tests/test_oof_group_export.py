import numpy as np
import pytest

from chem_repro.common.validation import regression_oof


def test_exported_membership_independently_certifies_scaffold_split():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    groups = np.repeat([f"scaffold-{i}" for i in range(10)], 3)
    result = regression_oof(x, x[:, 0] + .2*x[:, 1], groups,
                            [str(i) for i in range(30)], model="ElasticNet", bootstrap=100)
    rows = result["scaffold"]["oof"]
    assert len(rows) == 30 and len({r['observation_id'] for r in rows}) == 30
    for fold in range(5):
        train = {r['generic_scaffold_group'] for r in rows if r['fold'] != fold}
        test = {r['generic_scaffold_group'] for r in rows if r['fold'] == fold}
        assert test and not train.intersection(test)
    # Random folds retain group identities too; a real overlap must not be hidden.
    rows = result['random']['oof']
    assert any({r['generic_scaffold_group'] for r in rows if r['fold'] == f} &
               {r['generic_scaffold_group'] for r in rows if r['fold'] != f} for f in range(5))


def test_leaking_supplied_scaffold_folds_are_rejected():
    x = np.arange(60).reshape(30,2)
    groups = np.repeat(np.arange(10),3)
    with pytest.raises(ValueError, match='leak'):
        regression_oof(x, x[:,0], groups, list(range(30)), model='ElasticNet', bootstrap=100,
                       fold_assignments={s:np.arange(30)%5 for s in ['random','scaffold']})
