"""Open Syntelly-2024 recipe with declared public training data and new predictions.

This is method reproduction, not a replay of proprietary fitted weights. The two
fixed estimators are copied from the pinned existing Cannabis open recipe. Each
fold is fitted independently; no held-out data choose features, thresholds or weights.
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from chem_repro.common.applicability import ApplicabilityDomain, fingerprints


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(frame):
    molecules = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in frame.smiles]
    valid = np.array([m is not None and m.GetNumAtoms() > 0 for m in molecules])
    out = frame.loc[valid].copy()
    mols = [m for m, ok in zip(molecules, valid) if ok]
    out['canonical_smiles'] = [Chem.MolToSmiles(m, isomericSmiles=True) for m in mols]
    # Acyclic structures share the empty Murcko scaffold; do not quietly split it.
    out['scaffold'] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols]
    return out, mols, frame.loc[~valid].copy()


def metrics(y, p, task):
    if task == 'classification':
        return {'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                'average_precision': float(average_precision_score(y, p)),
                'positive_fraction_at_0_5': float(np.mean(p >= .5))}
    return {'rmse': float(np.sqrt(np.mean((y-p)**2))),
            'mae': float(mean_absolute_error(y, p)), 'r2': float(r2_score(y, p))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--recipe-root', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--task', choices=['classification', 'regression'], required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--cv', choices=['random', 'both'], default='both')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.recipe_root.resolve()))
    import tox_train as T

    raw_train, raw_query = pd.read_csv(args.train, dtype={'source_id':str}), pd.read_csv(args.query, dtype={'query_id':str})
    if not np.isfinite(raw_train.y.to_numpy(float)).all():
        raise ValueError('Training targets must be finite; no implicit label imputation')
    if args.task == 'classification' and set(raw_train.y.unique()) != {0, 1}:
        raise ValueError('Expected both binary classes')
    train, train_mols, rejected_train = parse(raw_train)
    query, query_mols, rejected_query = parse(raw_query)
    rejected_train.to_csv(args.output/'rejected_training.csv', index=False)
    rejected_query.to_csv(args.output/'rejected_query.csv', index=False)
    train.to_csv(args.output/'effective_training.csv', index=False)
    y = train.y.to_numpy(float)
    features = {'CatBoost_ECFP': T.featurize_ecfp(train_mols), 'XGBoost_fragments': T.featurize_fragments(train_mols)}
    query_features = {'CatBoost_ECFP': T.featurize_ecfp(query_mols), 'XGBoost_fragments': T.featurize_fragments(query_mols)}
    factories = {'CatBoost_ECFP': T.make_catboost, 'XGBoost_fragments': T.make_xgboost}
    spec = {'endpoint':args.endpoint, 'task':args.task, 'unit':args.unit,
            'training_sha256':digest(args.train), 'query_sha256':digest(args.query),
            'recipe_sha256':digest(args.recipe_root/'tox_train.py'),
            'worker_sha256':digest(__file__), 'domain_sha256':digest(Path(__file__).resolve().parents[1]/'common/applicability.py'),
            'protocol':'syntelly2024_open_two_boosters_v1', 'prediction_kind':'new_open_model_prediction',
            'n_train_raw':len(raw_train), 'n_train_valid':len(train),
            'n_query_raw':len(raw_query), 'n_query_valid':len(query),
            'fixed_ensemble':'arithmetic_mean_of_two_models', 'classification_threshold':.5,
            'software':{p:version(p) for p in ['numpy','pandas','scikit-learn','rdkit','catboost','xgboost']}}
    (args.output/'run_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    result = {'spec':spec, 'validation':{}, 'model_parameters':{}}
    arrays = {'y':y}
    folds = {}
    with threadpool_limits(limits=2):
        for scheme in (['identity_random','scaffold'] if args.cv == 'both' else ['identity_random']):
            groups = train.canonical_smiles.to_numpy() if scheme == 'identity_random' else train.scaffold.to_numpy()
            splitter = (StratifiedGroupKFold(5,shuffle=True,random_state=42) if args.task == 'classification'
                        else GroupKFold(5,shuffle=True,random_state=42))
            partitions = list(splitter.split(features['CatBoost_ECFP'],y,groups))
            folds[scheme] = []
            coverage = np.zeros(len(y),dtype=int)
            for index,(fit,held) in enumerate(partitions):
                if set(groups[fit]) & set(groups[held]):
                    raise ValueError('Training and validation groups overlap')
                coverage[held] += 1
                folds[scheme].append({'fold':index,'train_indices':fit.tolist(),'test_indices':held.tolist(),
                                      'group_overlap':0,'n_train':len(fit),'n_test':len(held)})
            if not np.all(coverage == 1):
                raise ValueError('OOF coverage is not exactly one prediction per record')
            ps = []
            result['validation'][scheme] = {}
            for name,factory in factories.items():
                p = np.full(len(y),np.nan)
                for fold,(fit,held) in enumerate(partitions):
                    model = factory(args.task)
                    model.set_params(**({'thread_count':2} if name.startswith('Cat') else {'n_jobs':2}))
                    model.fit(features[name][fit],y[fit])
                    p[held] = (model.predict_proba(features[name][held])[:,1] if args.task == 'classification'
                               else model.predict(features[name][held]))
                    print(json.dumps({'endpoint':args.endpoint,'scheme':scheme,'model':name,'completed_fold':fold}),flush=True)
                if not np.isfinite(p).all():
                    raise ValueError('Missing or nonfinite OOF predictions')
                ps.append(p);arrays[scheme+'__'+name] = p
                result['validation'][scheme][name] = metrics(y,p,args.task)
            blend = np.mean(ps,axis=0)
            arrays[scheme+'__mean'] = blend
            result['validation'][scheme]['mean'] = metrics(y,blend,args.task)
        for name,factory in factories.items():
            model=factory(args.task)
            model.set_params(**({'thread_count':2} if name.startswith('Cat') else {'n_jobs':2}))
            model.fit(features[name],y)
            query[name] = (model.predict_proba(query_features[name])[:,1] if args.task == 'classification'
                           else model.predict(query_features[name]))
            path=args.output/(name+('.cbm' if name.startswith('Cat') else '.json'))
            model.save_model(str(path))
            result['model_parameters'][name] = json.loads(json.dumps(model.get_params(),default=str).replace('NaN','"NaN"'))
    query['prediction'] = query[list(factories)].mean(axis=1)
    query['training_identity_overlap'] = query.canonical_smiles.isin(set(train.canonical_smiles))
    query['training_scaffold_overlap'] = query.scaffold.isin(set(train.scaffold))
    domain = ApplicabilityDomain.fit(fingerprints(train_mols))
    for name,values in domain.score(fingerprints(query_mols)).items():
        query[name] = values
    if args.unit == '-log10(mol/kg)':
        query['predicted_ld50_mg_kg'] = (10.0**(-query.prediction.to_numpy()) *
                                        np.asarray([Descriptors.MolWt(m) for m in query_mols]) * 1000)
    query.to_csv(args.output/'predictions.csv',index=False)
    np.savez_compressed(args.output/'oof_predictions.npz',**arrays)
    (args.output/'folds.json').write_text(json.dumps(folds)+'\n')
    result['domain'] = {'k':domain.k,'sigma':domain.sigma,'threshold':domain.threshold,
                        'reference_indices':domain.reference_indices.tolist(),
                        'reference_distances':domain.reference_distances.tolist(),
                        'interpretation':'relative training-domain score, not confidence or calibrated accuracy'}
    result['artifacts'] = {p.name:digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'endpoint':args.endpoint,'validation':result['validation'],'completed':True}),flush=True)


if __name__ == '__main__':
    main()
