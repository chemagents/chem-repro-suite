"""Observe the pinned upstream tox replay in an isolated dependency environment.

Original scientific functions are executed unchanged. A trace observer records their
OOF predictions/folds at return; only estimator thread counts are limited explicitly.
The source meta stage is not a strictly nested validation and is never labeled as one.
"""

import argparse
import hashlib
import importlib
import json
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from rdkit import RDLogger
from sklearn.model_selection import check_cv
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/cannabis_open_tox"
DATA = ROOT / "data/cannabis_open_tox"
ENDPOINTS = {
    "ames": "classification",
    "reproductive": "classification",
    "daphnia": "regression",
    "fathead": "regression",
}


def source_hashes():
    paths = [Path(__file__), *sorted(VENDOR.glob("*.py"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=list(ENDPOINTS), required=True)
    parser.add_argument("--scheme", choices=["random", "source_scaffold"], required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("Use one to four local CPU threads")
    if args.endpoint == "reproductive" and args.scheme != "random":
        raise ValueError("Supplement S3 has no reproductive scaffold reference; no silent substitute")
    args.output.mkdir(parents=True, exist_ok=True)
    expected = dict(
        line.strip().split("==")
        for line in (DATA / "worker_requirements.txt").read_text().splitlines()
        if line.strip()
    )
    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise RuntimeError("Worker dependency versions do not match the pinned source lock")
    manifest = json.loads((DATA / "manifest.json").read_text())
    csv = DATA / f"{args.endpoint}_aug.csv"
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert digest == manifest["files"][csv.name]["sha256"]
    initial_code = source_hashes()
    run_spec = {
        "cohort": "TOXRIC_plus_ECOTOX",
        "input_filename": csv.name,
        "endpoint": args.endpoint,
        "scheme": args.scheme,
        "threads": args.threads,
        "software": actual,
        "source_sha256": digest,
        "source_revision": manifest["revision"],
        "source_code_sha256": initial_code,
    }
    (args.output / "run_spec.json").write_text(json.dumps(run_spec, indent=2) + "\n")
    sys.path.insert(0, str(VENDOR))
    source = importlib.import_module("stack_v2" if args.scheme == "random" else "scaffold_eval")
    # Source scripts suppress all warnings. The observer restores them for the run log.
    warnings.resetwarnings()
    RDLogger.EnableLog("rdApp.warning")
    RDLogger.EnableLog("rdApp.error")
    original_bases = source.bases
    model_parameters = {}

    def controlled_bases(task):
        output = []
        for name, feature, factory in original_bases(task):

            def controlled(factory=factory, name=name):
                model = factory()
                params = model.get_params()
                if "thread_count" in params:
                    model.set_params(thread_count=args.threads)
                elif "n_jobs" in params:
                    model.set_params(n_jobs=args.threads)
                # Keep source parameters, including explicit missing-value sentinels as strings.
                model_parameters[name] = {
                    k: ("NaN" if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in model.get_params().items()
                }
                return model

            output.append((name, feature, controlled))
        return output

    source.bases = controlled_bases
    captured = {}
    last_progress = [None]

    def observer(frame, event, arg):
        if frame.f_code is not source.run.__code__:
            return None
        local = frame.f_locals
        completed = tuple(local.get("oof", {}))
        if completed != last_progress[0]:
            last_progress[0] = completed
            progress = {
                "endpoint": args.endpoint,
                "scheme": args.scheme,
                "completed_base_models": list(completed),
            }
            (args.output / "progress.json").write_text(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
        if event == "return":
            captured.update(
                {
                    k: local[k]
                    for k in ["y", "folds", "oof", "P", "meta", "bs", "mean_blend", "meta_s", "F", "n"]
                    if k in local
                }
            )
            if "groups" in local:
                captured["groups"] = local["groups"]
        return observer

    start = time.monotonic()
    sys.settrace(observer)
    try:
        with threadpool_limits(limits=args.threads):
            source.run(str(csv), ENDPOINTS[args.endpoint], args.endpoint)
    finally:
        sys.settrace(None)
    if not {"oof", "meta", "y", "folds", "meta_s"}.issubset(captured):
        raise RuntimeError("Source replay did not complete all required model stages")
    if source_hashes() != initial_code or hashlib.sha256(csv.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Code or inputs changed during replay")
    y = captured["y"]
    fold_ids = np.full(len(y), -1, int)
    folds = []
    for fold, (train, test) in enumerate(captured["folds"]):
        fold_ids[test] = fold
        folds.append({"fold": fold, "train_indices": train.tolist(), "test_indices": test.tolist()})
    meta_folds = []
    splitter = check_cv(5, y=y, classifier=ENDPOINTS[args.endpoint] == "classification")
    for fold, (train, test) in enumerate(splitter.split(captured["P"], y)):
        exposure = set()
        for base_fold in set(fold_ids[train]):
            exposure.update(test[fold_ids[test] != base_fold].tolist())
        groups = captured.get("groups")
        meta_folds.append(
            {
                "fold": fold,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "meta_test_labels_used_in_base_training_for_meta_train_features": len(exposure),
                "source_group_overlap": len(set(groups[train]) & set(groups[test]))
                if groups is not None
                else None,
            }
        )
    np.savez_compressed(
        args.output / "oof_predictions.npz",
        observed=y,
        base_fold=fold_ids,
        source_meta=captured["meta"],
        **captured["oof"],
    )
    (args.output / "folds.json").write_text(
        json.dumps({"base_folds": folds, "meta_folds": meta_folds}, indent=2) + "\n"
    )
    metrics = {
        "n_records": int(captured["n"]),
        "metric": "roc_auc" if ENDPOINTS[args.endpoint] == "classification" else "rmse",
        "base_scores": {k: float(v) for k, v in captured["bs"].items()},
        "mean_blend": float(captured["mean_blend"]),
        "source_meta_stack": float(captured["meta_s"]),
        "feature_dimensions": {k: int(v.shape[1]) for k, v in captured["F"].items()},
        "source_meta_validation": "replayed_non_nested_global_OOF_matrix",
        "acyclic_group_policy": "individual_canonical_molecule" if args.scheme == "source_scaffold" else None,
    }
    result = {
        "run_spec": run_spec,
        "metrics": metrics,
        "model_parameters": model_parameters,
        "seconds": time.monotonic() - start,
        "status": "completed_source_replay",
        "scientific_limitation": "Global base OOF features are reused for meta CV; this is not strict nested validation. Meta CV does not preserve scaffold groups.",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
