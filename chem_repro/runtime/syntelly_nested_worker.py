"""Declared public-data 2024 reconstruction with genuinely nested tuning.

This worker is separate from frozen v1. Unpublished dictionaries, historical folds,
Optuna trial counts and author weights are not reconstructed by fitting new models.
Run from an immutable snapshot with a JSON configuration and a pinned environment.
"""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from chem_repro.cannabis.paper2024 import (
    ALLOWED_ELEMENTS,
    ASSUMPTIONS,
    catboost_search_parameters,
    descriptor_frame,
    prepare,
    xgboost_search_space,
)
from chem_repro.common.applicability import ApplicabilityDomain, fingerprints
from chem_repro.common.fragment_selection import FragmentSelector

REQUIRED_SOFTWARE = {"numpy", "pandas", "scikit-learn", "rdkit", "catboost", "xgboost", "optuna", "joblib", "threadpoolctl"}
FULL_GRID_ENDPOINTS = {"mouse_oral_ld50", "mouse_intraperitoneal_ld50", "mouse_intravenous_ld50", "bbb"}

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def scores(y, p, task):
    if task == "classification":
        return {"roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                "average_precision": float(average_precision_score(y, p)) if np.any(y == 1) else None}
    return {"rmse": float(np.sqrt(np.mean((y - p)**2))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None}


def partitions(y, groups, task, n_splits, seed):
    """Never reduce the requested folds, drop a fold or split a scaffold silently."""
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError("Insufficient groups for the declared number of folds")
    splitter = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                if task == "classification" else GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    folds = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    counts = np.zeros(len(y), dtype=int)
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Grouping leaked between training and validation")
        if task == "classification" and len(np.unique(y[train])) != 2:
            raise ValueError("A declared training fold contains only one class; no silent split replacement")
        counts[test] += 1
    if not np.all(counts == 1):
        raise ValueError("Every row must occur in exactly one held-out partition")
    return folds


def nested_partitions(y, groups, task, outer_folds, inner_folds, seed):
    outer = partitions(y, groups, task, outer_folds, seed)
    result = []
    for fold, (train, test) in enumerate(outer):
        inner = partitions(y[train], groups[train], task, inner_folds, seed + fold + 1)
        global_inner = [(train[a], train[b]) for a, b in inner]
        for a, b in global_inner:
            if set(test) & (set(a) | set(b)) or set(groups[a]) & set(groups[b]):
                raise ValueError("Outer test information entered inner selection")
        result.append((train, test, global_inner))
    return result


def fingerprint_array(molecules):
    bits = fingerprints(molecules)
    matrix = np.zeros((len(bits), 2048), dtype=np.uint8)
    for i, fp in enumerate(bits):
        DataStructs.ConvertToNumpyArray(fp, matrix[i])
    return matrix, bits


def model(family, task, parameters, seed, threads):
    if family == "CatBoost_ECFP":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
        return cls(**parameters, loss_function="Logloss" if task == "classification" else "RMSE",
                   random_seed=seed, thread_count=threads, verbose=False, allow_writing_files=False)
    from xgboost import XGBClassifier, XGBRegressor

    cls = XGBClassifier if task == "classification" else XGBRegressor
    return cls(**parameters, random_state=seed, n_jobs=threads, tree_method="hist",
               eval_metric="logloss" if task == "classification" else "rmse")


def predict(estimator, x, task):
    if task == "classification":
        column = list(estimator.classes_).index(1)
        return estimator.predict_proba(x)[:, column].astype(float)
    return estimator.predict(x).astype(float)


def fitted_features(raw, fragment_names, train, test):
    selector = FragmentSelector(fragment_names).fit(raw.iloc[train])
    return selector, selector.transform(raw.iloc[train]), selector.transform(raw.iloc[test])


def tune(family, task, data, y, fragments, outer_train, inner, config, path, seed):
    """Objective is pooled inner OOF performance; outer test never enters this function."""
    inner_features = []
    lineage = []
    for train, valid in inner:
        if family == "XGBoost_fragments":
            selector, xtrain, xvalid = fitted_features(data, fragments, train, valid)
            names = selector.selected_features_
        else:
            xtrain, xvalid, names = data[train], data[valid], None
        inner_features.append((train, valid, xtrain, xvalid))
        lineage.append({"training_indices": train.tolist(), "validation_indices": valid.tolist(),
                        "selected_features": names})
    write_json(path / f"{family}_inner_lineage.json", lineage)
    trials = []

    def objective(parameters, number):
        predictions = np.full(len(y), np.nan)
        per_fold = []
        emit("candidate_start", family=family, candidate=number, scope=path.name)
        start = time.monotonic()
        for fold, (train, valid, xtrain, xvalid) in enumerate(inner_features):
            estimator = model(family, task, parameters, seed + fold, config["threads"])
            estimator.fit(xtrain, y[train])
            predictions[valid] = predict(estimator, xvalid, task)
            per_fold.append(scores(y[valid], predictions[valid], task))
        if not np.isfinite(predictions[outer_train]).all():
            raise ValueError("Incomplete inner OOF prediction coverage")
        values = scores(y[outer_train], predictions[outer_train], task)
        value = values["roc_auc" if task == "classification" else "rmse"]
        if value is None:
            raise ValueError("Pooled inner selection metric is undefined")
        trials.append({"number": number, "parameters": parameters, "pooled_inner_metrics": values,
                       "inner_fold_metrics": per_fold, "seconds": time.monotonic() - start})
        write_json(path / f"{family}_completed_candidates.json", trials)
        emit("candidate_complete", family=family, candidate=number, objective=value, scope=path.name)
        return value

    if family == "CatBoost_ECFP":
        import optuna
        from catboost import CatBoostError

        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=config["optuna_startup_trials"])
        study = optuna.create_study(direction="maximize" if task == "classification" else "minimize",
                                    sampler=sampler, pruner=optuna.pruners.NopPruner())

        def optuna_objective(trial):
            parameters = catboost_search_parameters(trial)
            try:
                return objective(parameters, trial.number)
            except CatBoostError as exc:
                trial.set_user_attr("backend_failure", str(exc))
                raise

        study.optimize(optuna_objective, n_trials=config["optuna_trials"], n_jobs=1, catch=(CatBoostError,))
        write_json(path / "optuna_trials.json", [
            {"number": t.number, "state": t.state.name, "value": t.value, "parameters": t.params,
             "user_attrs": t.user_attrs} for t in study.trials])
        best = study.best_trial
        parameters = dict(best.params)
    else:
        full = config["endpoint"] in FULL_GRID_ENDPOINTS
        grid = xgboost_search_space(full=full)
        if len(grid) > config["max_xgboost_candidates"]:
            raise ValueError(f"Declared grid has {len(grid)} candidates, above configured budget; no truncation")
        values = [objective(p, i) for i, p in enumerate(grid)]
        best = int(np.argmax(values) if task == "classification" else np.argmin(values))
        parameters = dict(grid[best])
    write_json(path / f"{family}_selected_parameters.json", parameters)
    return parameters


def run(config):
    if config["outer_folds"] != 5 or config["inner_folds"] < 2 or config["optuna_trials"] < 1:
        raise ValueError("Require five outer folds and a nonempty declared inner search")
    if not 1 <= config["threads"] <= 4:
        raise ValueError("Declared local CPU budget is1–4threads")
    if config["optuna_startup_trials"] > config["optuna_trials"]:
        raise ValueError("Startup trials exceed total search budget")
    if not REQUIRED_SOFTWARE <= set(config["software"]):
        raise ValueError("Scientific dependency versions must be pinned")
    grid_size = len(xgboost_search_space(full=config["endpoint"] in FULL_GRID_ENDPOINTS))
    if grid_size > config["max_xgboost_candidates"]:
        raise ValueError(f"Declared grid has {grid_size} candidates, above configured budget; no truncation")
    for key in ["training", "query"]:
        if digest(config[key + "_path"]) != config[key + "_sha256"]:
            raise ValueError(f"{key} input digest mismatch")
    software = {name: version(name) for name in config["software"]}
    if software != config["software"]:
        raise ValueError("Pinned scientific environment mismatch")
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    source_root = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(source_root)): digest(p) for p in source_root.rglob("*.py")}
    write_json(root / "code_hashes.json", code)
    raw = pd.read_csv(config["training_path"], dtype={"source_id": str}, keep_default_na=False)
    prepared = prepare(raw, config["task"], config["endpoint"] + ":public_collection", config["regression_aggregation"])
    training = prepared.molecules.copy()
    prepared.records.to_csv(root / "training_record_ledger.csv", index=False)
    training.to_csv(root / "prepared_training.csv", index=False)
    query = pd.read_csv(config["query_path"], dtype={"query_id": str}, keep_default_na=False)
    molecules = [Chem.MolFromSmiles(s) for s in training.smiles]
    query_molecules = [Chem.MolFromSmiles(s) for s in query.smiles]
    if any(m is None or not m.GetNumAtoms() for m in query_molecules):
        raise ValueError("Query contains invalid structures; preserve and review before prediction")
    for molecule in query_molecules:
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
    identities = training.smiles.to_numpy(str)
    scaffolds = np.asarray([MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in molecules])
    y = training.y.to_numpy(float)
    layouts = {scheme: nested_partitions(y, identities if scheme == "identity_random" else scaffolds,
                config["task"], config["outer_folds"], config["inner_folds"], config["seed"])
               for scheme in config["schemes"]}
    if set(config["schemes"]) != {"identity_random", "scaffold"} or len(config["schemes"]) != 2:
        raise ValueError("Both declared validation schemes are required, without duplicates")
    for scheme, folds in layouts.items():
        write_json(root / f"{scheme}_partitions.json", [
            {"train": a.tolist(), "test": b.tolist(), "inner": [{"train": c.tolist(), "test": d.tolist()} for c, d in inner]}
            for a, b, inner in folds])
    xfp, bits = fingerprint_array(molecules)
    qfp, qbits = fingerprint_array(query_molecules)
    xfrag, fragments = descriptor_frame(training.smiles)
    qfrag, query_fragments = descriptor_frame(query.smiles)
    if fragments != query_fragments or list(xfrag) != list(qfrag):
        raise ValueError("Query descriptor schema differs from training")
    write_json(root / "descriptor_dictionary.json", {"columns": list(xfrag), "fragments": fragments,
                                                   "scope": "RDKit substitute; author dictionary unavailable"})
    domain = ApplicabilityDomain.fit(bits).score(qbits)
    families = ["CatBoost_ECFP", "XGBoost_fragments"]
    arrays, validation, fold_metrics, roundtrip = {"y": y}, {}, [], []
    emit("prepared", endpoint=config["endpoint"], preparation=prepared.summary, software=software, config=config)
    with threadpool_limits(limits=config["threads"]):
        for scheme, folds in layouts.items():
            oof = {name: np.full(len(y), np.nan) for name in families}
            query_predictions = {name: [] for name in families}
            for fold, (train, test, inner) in enumerate(folds):
                path = root / f"{scheme}_fold{fold}"
                path.mkdir()
                for family in families:
                    data = xfp if family == "CatBoost_ECFP" else xfrag
                    parameters = tune(family, config["task"], data, y, fragments, train, inner, config,
                                      path, config["seed"] + fold * 100)
                    if family == "XGBoost_fragments":
                        selector, xt, xv = fitted_features(data, fragments, train, test)
                        xq = selector.transform(qfrag)
                        selector.selection_audit_.to_csv(path / "outer_feature_selection.csv", index=False)
                    else:
                        selector, xt, xv, xq = None, xfp[train], xfp[test], qfp
                    fitted = model(family, config["task"], parameters, config["seed"] + fold, config["threads"])
                    fitted.fit(xt, y[train])
                    p, q = predict(fitted, xv, config["task"]), predict(fitted, xq, config["task"])
                    artifact = path / f"{family}.joblib"
                    joblib.dump({"model": fitted, "selector": selector, "feature_columns": list(xfrag),
                                 "family": family, "task": config["task"]}, artifact, compress=3)
                    restored = joblib.load(artifact)
                    if family == "XGBoost_fragments":
                        xv2, xq2 = restored["selector"].transform(xfrag.iloc[test]), restored["selector"].transform(qfrag)
                    else:
                        xv2, xq2 = xv, xq
                    error = max(float(np.max(np.abs(predict(restored["model"], xv2, config["task"]) - p))),
                                float(np.max(np.abs(predict(restored["model"], xq2, config["task"]) - q))))
                    if error > 1e-12:
                        raise ValueError("Serialized model predictions changed")
                    roundtrip.append({"scheme": scheme, "fold": fold, "family": family,
                                      "max_absolute_error": error, "test_rows": len(test), "query_rows": len(query)})
                    oof[family][test] = p
                    query_predictions[family].append(q)
                    fold_metrics.append({"scheme": scheme, "fold": fold, "family": family,
                                         "n_train": len(train), "n_test": len(test), **scores(y[test], p, config["task"])})
                    emit("outer_fit_complete", **fold_metrics[-1])
            oof["fixed_mean_blend"] = (oof[families[0]] + oof[families[1]]) / 2
            if not all(np.isfinite(values).all() for values in oof.values()):
                raise ValueError("Outer OOF coverage is incomplete")
            validation[scheme] = {name: scores(y, predictions, config["task"]) for name, predictions in oof.items()}
            for name, predictions in oof.items():
                arrays[scheme + "__" + name] = predictions
            out = query.copy()
            for family in families:
                predictions = np.asarray(query_predictions[family])
                arrays[scheme + "__query__" + family] = predictions
                out[family] = predictions.mean(axis=0)
            out["prediction"] = (out[families[0]] + out[families[1]]) / 2
            for name, values in domain.items():
                out[name] = values
            out["unit"] = config["unit"]
            canonical = [Chem.MolToSmiles(m, isomericSmiles=True) for m in query_molecules]
            identity_set, scaffold_set = set(identities), set(scaffolds)
            out["canonical_smiles"] = canonical
            out["training_identity_overlap"] = [s in identity_set for s in canonical]
            out["training_scaffold_overlap"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) in scaffold_set for m in query_molecules]
            out["matches_training_element_policy"] = [
                any(a.GetSymbol() == "C" for a in m.GetAtoms()) and
                all(a.GetSymbol() in set(ALLOWED_ELEMENTS) | {"H"} for a in m.GetAtoms())
                for m in query_molecules]
            out.to_csv(root / f"{scheme}_query_predictions.csv", index=False)
            emit("scheme_complete", scheme=scheme, validation=validation[scheme])
    np.savez_compressed(root / "oof_and_query_predictions.npz", **arrays)
    pd.DataFrame(fold_metrics).to_csv(root / "fold_metrics.csv", index=False)
    write_json(root / "model_roundtrips.json", roundtrip)
    artifacts = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    receipt = {"protocol": "paper2024_public_nested_v2", "status": "completed_declared_reconstruction",
               "config": config, "software": software, "preparation": prepared.summary,
               "assumptions": ASSUMPTIONS, "validation": validation, "model_roundtrips": len(roundtrip),
               "feature_selection_scope": "Refitted only inside inner and outer training folds",
               "query_predictor": "Mean of outer-fold models, fixed arithmetic mean of two families",
               "domain_scope": "Full prepared training reference, relative kNN score, not calibrated accuracy",
               "original_author_weights": False, "historical_optimum_or_convergence_claim": False,
               "artifacts": artifacts}
    write_json(root / "result.json", receipt)
    emit("completed", endpoint=config["endpoint"], validation=validation, model_roundtrips=len(roundtrip))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
