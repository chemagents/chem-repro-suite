"""Figure 3 ten-fold validation, separate from the deposited repeated holdouts.

Fold count is specified by the article. Stratification, shuffling and seed are
explicit reconstruction choices because the author memberships are unavailable.
"""
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..common.artifacts import artifact_root, save_artifact, save_json
from ..common.classification import binary_metrics
from ..common.contracts import EvidenceNote
from ..common.datasets import table, verified_manifest
from ..common.rmt import fit_residue_selector
from .analysis import result, metric
from .data import TARGETS, load, residue_matrix


def partitions(frame, *, seed=42):
    labeled = frame.loc[frame.activity.notna()].reset_index(drop=True)
    if labeled.ligand_id.isna().any() or labeled.ligand_id.duplicated().any():
        raise ValueError('Each labeled source record needs one unique ID')
    y = labeled.activity.to_numpy(int)
    if set(y) != {0, 1} or min(np.bincount(y)) < 10:
        raise ValueError('Ten-fold stratification requires at least ten records per class')
    return [(labeled.iloc[tr].copy(), labeled.iloc[te].copy()) for tr, te in
            StratifiedKFold(10, shuffle=True, random_state=seed).split(labeled, y)]


def fit_fold(train, test, *, index, iterations, matrices, include_models=False):
    """All selection and scaling are fitted only on the supplied training rows."""
    if set(train.ligand_id) & set(test.ligand_id):
        raise ValueError('Training and test IDs overlap')
    y_train, y_test = train.activity.to_numpy(int), test.activity.to_numpy(int)
    registry = {'fold': index, 'train_ids': train.ligand_id.astype(int).tolist(),
                'test_ids': test.ligand_id.astype(int).tolist(),
                'train_labels': y_train.tolist(), 'test_labels': y_test.tolist()}
    selected, selections, predictions, metrics, fitted = {}, [], [], [], {}
    for target, matrix in matrices.items():
        x_train = matrix.loc[train.ligand_id].to_numpy(float)
        x_test = matrix.loc[test.ligand_id].to_numpy(float)
        selector = fit_residue_selector(x_train, y_train, list(matrix),
                                      inner_splits=5, inner_test_size=.2, seed=42)
        names = [selector.feature_names[i] for i in selector.selected_indices]
        selected[target] = (selector.transform(x_train, list(matrix)),
                            selector.transform(x_test, list(matrix)),
                            [target + ':' + name for name in names])
        selections.append({'fold': index, 'target': target, **selector.diagnostics,
                           'selected_features': names,
                           'ranked_residues': table(selector.ranking, 'Training-only selection').rows,
                           'inner_scan': table(selector.scan, 'Training-only scan').rows})
    selected['ALL6'] = (np.column_stack([x[0] for x in selected.values()]),
                        np.column_stack([x[1] for x in selected.values()]),
                        [n for x in selected.values() for n in x[2]])
    for target, (x_train, x_test, names) in selected.items():
        estimators = {
            'CatBoost': CatBoostClassifier(iterations=iterations, depth=6, learning_rate=.05,
                loss_function='Logloss', random_seed=42, thread_count=2, verbose=False,
                allow_writing_files=False, use_best_model=False, task_type='CPU'),
            'LinearSVM': make_pipeline(StandardScaler(), LinearSVC(C=1., class_weight='balanced',
                dual='auto', max_iter=20000, random_state=42)),
        }
        for model, estimator in estimators.items():
            estimator.fit(x_train, y_train)
            probability = model == 'CatBoost'
            values = estimator.predict_proba(x_test)[:, 1] if probability else estimator.decision_function(x_test)
            kind, threshold = ('probability', .5) if probability else ('decision_margin', 0.)
            metrics.append({'fold': index, 'target': target, 'model': model,
                            'n_selected_features': len(names),
                            **binary_metrics(y_test, values, threshold=threshold, score_kind=kind)})
            predictions.extend({'fold': index, 'target': target, 'model': model, 'ligand_id': int(lid),
                                'label': int(y), 'score': float(s), 'score_kind': kind}
                               for lid, y, s in zip(test.ligand_id, y_test, values, strict=True))
            if include_models:
                if probability:
                    # CatBoost's JSON serializer writes a file; retain its actual fitted bytes.
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'model.json'
                        estimator.save_model(str(path), format='json')
                        model_data = json.loads(path.read_text())
                else:
                    scaler, svc = estimator.steps[0][1], estimator.steps[1][1]
                    model_data = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                                  'coef': svc.coef_.tolist(), 'intercept': svc.intercept_.tolist(),
                                  'classes': svc.classes_.tolist()}
                fitted[target + '/' + model] = {'features': names, 'model': model_data}
    return {'registry': registry, 'selections': selections, 'metrics': metrics,
            'predictions': predictions, 'models': fitted}


def validate_paper_cv(split_indices=None, *, iterations=500, include_models=False, seed=42):
    indices = list(range(10)) if split_indices is None else split_indices
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < 10 for i in indices):
        raise ValueError('Paper ten-fold indices must be unique integers in 0..9')
    if type(iterations) is not int or not 10 <= iterations <= 1000:
        raise ValueError('Use 10..1000 CatBoost iterations')
    pairs = partitions(load(), seed=seed)
    package = Path(__file__).resolve().parents[1]
    code = {n: hashlib.sha256((package / n).read_bytes()).hexdigest() for n in
            ['cannabis/paper_cv.py', 'cannabis/data.py', 'common/rmt.py', 'common/classification.py']}
    request = {'protocol': 'paper_kfold10', 'fold_seed': seed, 'iterations': iterations,
               'include_models': include_models, 'code': code,
               'data': verified_manifest('cannabis')['files'],
               'software': {n: version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost']}}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    root = Path(os.getenv('CHEM_REPRO_PAPER_RTE_RESULTS_DIR', str(artifact_root() / 'paper_rte_runs'))) / key
    root.mkdir(parents=True, exist_ok=True)
    matrices = {target: residue_matrix(target) for target in TARGETS}
    runs, reused = [], True
    import fcntl
    with (root / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('The same paper CV is already running') from exc
        for index in indices:
            path = root / f'fold_{index:02d}.json'
            train, test = pairs[index]
            if path.exists():
                envelope = json.loads(path.read_text())
                payload = envelope['result']
                if (envelope['request'] != request or
                    envelope['sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() or
                    payload['registry']['train_ids'] != train.ligand_id.astype(int).tolist() or
                    payload['registry']['test_ids'] != test.ligand_id.astype(int).tolist() or
                    payload['registry']['train_labels'] != train.activity.astype(int).tolist() or
                    payload['registry']['test_labels'] != test.activity.astype(int).tolist()):
                    raise ValueError('Paper CV cache provenance or content differs')
            else:
                reused = False
                payload = fit_fold(train, test, index=index, iterations=iterations, matrices=matrices,
                                   include_models=include_models)
                envelope = {'request': request, 'result': payload,
                            'sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(envelope, allow_nan=False))
                tmp.replace(path)
            runs.append(payload)
    predictions = pd.DataFrame([r for run in runs for r in run['predictions']])
    rows = []
    # Recompute all displayed metrics from complete held-out scores, including cache hits.
    for (fold, target, model), g in predictions.groupby(['fold', 'target', 'model']):
        rows.append({'fold': int(fold), 'target': target, 'model': model,
                     **binary_metrics(g.label, g.score, threshold=.5 if model == 'CatBoost' else 0.,
                                      score_kind='probability' if model == 'CatBoost' else 'decision_margin')})
    scores = pd.DataFrame(rows)
    summary = scores.groupby(['target', 'model']).agg(n_folds=('fold', 'nunique'),
        mean_roc_auc=('roc_auc', 'mean'), std_roc_auc=('roc_auc', 'std'),
        mean_pr_auc_trapezoid=('pr_auc_trapezoidal', 'mean'), std_pr_auc_trapezoid=('pr_auc_trapezoidal', 'std'),
        mean_average_precision=('average_precision', 'mean'), std_average_precision=('average_precision', 'std'),
        mean_fpr=('fpr', 'mean'), std_fpr=('fpr', 'std'),
        mean_recall=('recall', 'mean'), std_recall=('recall', 'std')).reset_index()
    pooled = []
    n_labeled = len(load().loc[load().activity.notna()])
    if set(indices) == set(range(10)):
        for (target, model), g in predictions.groupby(['target', 'model']):
            if len(g) != n_labeled or g.ligand_id.duplicated().any():
                raise ValueError('Complete ten-fold OOF must score every source record exactly once')
            pooled.append({'target': target, 'model': model, 'n_records': len(g),
                **binary_metrics(g.label, g.score, threshold=.5 if model == 'CatBoost' else 0.,
                                 score_kind='probability' if model == 'CatBoost' else 'decision_margin')})
    selections = [{k: v for k, v in r.items() if k not in {'ranked_residues', 'inner_scan'}}
                  for run in runs for r in run['selections']]
    output = result('Executed Figure 3 ten-fold RTE validation on the available labeled source cohort. '
        'Each residue selector and scaler was fitted inside its outer training partition. '
        'Mean and sample SD across folds are separate from pooled OOF.',
        [metric('completed_model_fits', len(scores)), metric('completed_outer_folds', len(indices)),
         metric('labeled_source_records', n_labeled)],
        {'fold_metrics': table(scores, 'Metrics recomputed from held-out predictions'),
         'fold_summary': table(summary, 'Ten-fold means and sample SD; distinct from repeated holdout'),
         'pooled_oof': table(pd.DataFrame(pooled), 'Only present for all ten disjoint test folds'),
         'residue_selections': table(pd.DataFrame(selections), 'Training-only Appendix A selections')},
        'cannabis.rte.paper_kfold10.v1', notes=[EvidenceNote(code='PAPER_FOLD_MEMBERSHIP_UNAVAILABLE',
        message='Figure 3 specifies 10-fold CV. The author fold membership, seed and exact classifier '
        'hyperparameters are not provided. This explicit stratified ten-fold reconstruction uses seed42, '
        'CatBoost depth6/learning_rate0.05 and balanced linear SVM C1. It does not silently substitute '
        'the deposited 30 repeated 80/20 holdouts. No settings are selected by closeness to paper metrics.')],
        outer_protocol='paper_kfold10', indices=indices, fold_seed=seed, iterations=iterations,
        inner_splits=5, inner_test_size=.2, inner_seed=42, fold_aggregation='unweighted mean and sample SD',
        outer_test_used_in_fit=False)
    output.metadata.reused_from_cache = reused
    base = os.getenv('CHEM_REPRO_PUBLIC_BASE_URL', 'http://127.0.0.1:7341')
    output.artifacts.append(save_artifact(predictions.to_csv(index=False).encode(), suffix='csv',
        media_type='text/csv', public_base_url=base, display_name='paper_rte_tenfold_heldout_predictions.csv'))
    output.artifacts.append(save_json({'request': request, 'folds': [
        {k: v for k, v in run.items() if k not in {'predictions', 'models'}} for run in runs]},
        public_base_url=base, display_name='paper_rte_tenfold_selection_and_partition_audit.json'))
    if include_models:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for run in runs:
                z.writestr(f"fold_{run['registry']['fold']:02d}/models.json", json.dumps(run['models']))
        output.artifacts.append(save_artifact(buf.getvalue(), suffix='zip', media_type='application/zip',
            public_base_url=base, display_name='paper_rte_tenfold_models.zip'))
    from ..common.figures import initialize_style, export_figure
    import matplotlib.pyplot as plt
    initialize_style()
    fig, ax = plt.subplots(figsize=(6.75, 3.4), layout='constrained')
    targets = list(TARGETS) + ['ALL6']
    for offset, model, color in [(-.16, 'CatBoost', '#0072B2'), (.16, 'LinearSVM', '#D55E00')]:
        g = summary.loc[summary.model == model].set_index('target').loc[targets]
        ax.errorbar(np.arange(len(targets)) + offset, g.mean_roc_auc,
                    yerr=g.std_roc_auc.fillna(0), fmt='o', capsize=3, label=model, color=color)
    ax.set(xticks=range(len(targets)), xticklabels=targets, ylabel='ROC-AUC', xlabel='Target descriptor panel')
    ax.legend(frameon=False)
    output.artifacts.extend(export_figure(fig, 'paper_rte_tenfold_mean_sd', port=7341))
    return output
