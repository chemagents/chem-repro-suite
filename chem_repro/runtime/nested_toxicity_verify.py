"""Independently replay completed nested-model predictions without fitting models."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem_repro.cannabis.paper2024 import descriptor_frame, prepare
from chem_repro.runtime.syntelly_nested_worker import digest, fingerprint_array, predict, scores, write_json


def verify(result_dir, source_dir):
    root, source = Path(result_dir).resolve(), Path(source_dir).resolve()
    receipt = json.loads((root / "result.json").read_text())
    if receipt["protocol"] != "paper2024_public_nested_v2" or receipt["status"] != "completed_declared_reconstruction":
        raise ValueError("Require a completed nested-v2 result")
    config = receipt["config"]
    if {name: version(name) for name in receipt["software"]} != receipt["software"]:
        raise ValueError("Verification environment differs from model environment")
    for name, expected in receipt["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("Result artifact changed or escaped the result directory")
    code = json.loads((root / "code_hashes.json").read_text())
    package_root = Path(__file__).resolve().parents[1]
    for name, expected in code.items():
        path = (package_root / name).resolve()
        if not path.is_relative_to(package_root) or digest(path) != expected:
            raise ValueError("Loaded scientific implementation differs from the fitted snapshot")
    for key in ["training", "query"]:
        if digest(source / config[key + "_path"]) != config[key + "_sha256"]:
            raise ValueError("Original model input checksum mismatch")
    raw = pd.read_csv(source / config["training_path"], dtype={"source_id": str}, keep_default_na=False)
    prepared = prepare(raw, config["task"], config["endpoint"] + ":public_collection", config["regression_aggregation"])
    training = pd.read_csv(root / "prepared_training.csv", keep_default_na=False)
    if prepared.summary != receipt["preparation"] or list(training.smiles) != list(prepared.molecules.smiles):
        raise ValueError("Prepared cohort differs from replayed source preparation")
    np.testing.assert_allclose(training.y, prepared.molecules.y, rtol=0, atol=1e-14)
    y = training.y.to_numpy(float)
    query = pd.read_csv(source / config["query_path"], dtype={"query_id": str}, keep_default_na=False)
    mols = [Chem.MolFromSmiles(s) for s in training.smiles]
    qmols = [Chem.MolFromSmiles(s) for s in query.smiles]
    for m in qmols:
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
    xfp, _ = fingerprint_array(mols)
    qfp, _ = fingerprint_array(qmols)
    xfrag, _ = descriptor_frame(training.smiles)
    qfrag, _ = descriptor_frame(query.smiles)
    scaffolds = np.asarray([MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols])
    families = ["CatBoost_ECFP", "XGBoost_fragments"]
    checked, fold_audit, validation = [], [], {}
    with np.load(root / "oof_and_query_predictions.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["y"], y, rtol=0, atol=0)
        for scheme in config["schemes"]:
            groups = training.smiles.to_numpy(str) if scheme == "identity_random" else scaffolds
            counts = np.zeros(len(y), dtype=int)
            folds = json.loads((root / f"{scheme}_partitions.json").read_text())
            if len(folds) != 5:
                raise ValueError("Outer fold coverage is not five")
            for fold, definition in enumerate(folds):
                train, test = np.asarray(definition["train"]), np.asarray(definition["test"])
                if set(train) & set(test) or set(train) | set(test) != set(range(len(y))):
                    raise ValueError("Outer fold partition is invalid")
                if set(groups[train]) & set(groups[test]):
                    raise ValueError("Outer grouping leakage")
                inner_counts = np.zeros(len(y), dtype=int)
                if len(definition["inner"]) != config["inner_folds"]:
                    raise ValueError("Inner fold coverage changed")
                for part in definition["inner"]:
                    a, b = np.asarray(part["train"]), np.asarray(part["test"])
                    if set(a) & set(b) or set(a) | set(b) != set(train) or set(groups[a]) & set(groups[b]):
                        raise ValueError("Inner or outer-test leakage")
                    inner_counts[b] += 1
                if not np.all(inner_counts[train] == 1) or np.any(inner_counts[test]):
                    raise ValueError("Inner OOF coverage is incomplete")
                counts[test] += 1
                path = root / f"{scheme}_fold{fold}"
                for family in families:
                    # Files are trusted local run outputs, checksum-verified above
                    # before any model deserialization.
                    saved = joblib.load(path / f"{family}.joblib")
                    if family == "XGBoost_fragments":
                        selector = saved["selector"]
                        if selector.n_training_records_ != len(train):
                            raise ValueError("Outer selector training size mismatch")
                        xv, xq = selector.transform(xfrag.iloc[test]), selector.transform(qfrag)
                    else:
                        xv, xq = xfp[test], qfp
                    p, q = predict(saved["model"], xv, config["task"]), predict(saved["model"], xq, config["task"])
                    expected_p = arrays[scheme + "__" + family][test]
                    expected_q = arrays[scheme + "__query__" + family][fold]
                    error = max(float(np.max(np.abs(p - expected_p))), float(np.max(np.abs(q - expected_q))))
                    if error > 1e-12:
                        raise ValueError("Saved model does not reproduce OOF/query predictions")
                    checked.append({"scheme": scheme, "fold": fold, "family": family,
                                    "max_absolute_error": error, "n_test": len(test), "n_query": len(query)})
                fold_audit.append({"scheme": scheme, "fold": fold, "n_train": len(train), "n_test": len(test),
                                   "outer_group_overlap": 0, "inner_group_overlap": 0,
                                   "outer_test_rows_in_inner_search": 0})
            if not np.all(counts == 1):
                raise ValueError("Outer OOF coverage is incomplete")
            blend = (arrays[scheme + "__" + families[0]] + arrays[scheme + "__" + families[1]]) / 2
            np.testing.assert_allclose(arrays[scheme + "__fixed_mean_blend"], blend, rtol=0, atol=1e-14)
            validation[scheme] = {}
            for family in families + ["fixed_mean_blend"]:
                computed = scores(y, arrays[scheme + "__" + family], config["task"])
                for metric, value in computed.items():
                    expected = receipt["validation"][scheme][family][metric]
                    if value is None:
                        if expected is not None:
                            raise ValueError("Undefined metric was replaced by a number")
                    elif expected is None or abs(value - expected) > 1e-12:
                        raise ValueError("Stored validation metric disagrees with OOF predictions")
                validation[scheme][family] = computed
            predictions = pd.read_csv(root / f"{scheme}_query_predictions.csv", dtype={"query_id": str})
            if list(predictions.query_id) != list(query.query_id) or list(predictions.smiles) != list(query.smiles):
                raise ValueError("Query identity/order changed")
            for family in families:
                np.testing.assert_allclose(predictions[family], arrays[scheme + "__query__" + family].mean(axis=0), rtol=0, atol=1e-14)
            np.testing.assert_allclose(predictions.prediction, (predictions[families[0]] + predictions[families[1]]) / 2, rtol=0, atol=1e-14)
    return {"result_sha256": digest(root / "result.json"), "verifier_sha256": digest(__file__),
            "model_replays": checked, "fold_audit": fold_audit, "validation": validation,
            "fits_performed": 0, "source_preparation_rechecked": True,
            "interpretation": "Verified declared reconstruction; not original author weights or independent biological validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.result_dir, args.source_dir)
    write_json(args.report, result)
    print(json.dumps({"model_replays": len(result["model_replays"]), "folds": len(result["fold_audit"]),
                      "validation": result["validation"]}), flush=True)
