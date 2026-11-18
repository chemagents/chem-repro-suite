"""Fit an RTE branch on the same outer partitions as the actual structural models."""

import fcntl
import hashlib
import io
import json
import os
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..common.artifacts import export_tables, save_artifact
from ..common.classification import evaluate_product
from ..common.contracts import EvidenceNote
from ..common.datasets import table, verified_manifest
from ..common.rmt import fit_residue_selector
from .analysis import metric, result
from .data import TARGETS, load, residue_matrix
from .structure_validation import validate_structure


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    body = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body)
    temporary.replace(path)


def aligned_scores(structural, rte, *, expected_ids):
    """Reject absent, duplicated or inconsistent identities before multiplying any scores."""
    required = {"ligand_id", "activity", "probability"}
    for frame in (structural, rte):
        if not required.issubset(frame.columns) or frame.ligand_id.isna().any() or frame.ligand_id.duplicated().any():
            raise ValueError("One score per nonmissing ligand ID is required in both branches")
        if set(frame.ligand_id) != set(expected_ids):
            raise ValueError("Both branches must cover exactly the declared held-out population")
    a = structural.set_index("ligand_id").loc[expected_ids]
    b = rte.set_index("ligand_id").loc[expected_ids]
    if not np.array_equal(a.activity.to_numpy(), b.activity.to_numpy()):
        raise ValueError("Branch labels disagree for the same ligand IDs")
    # Also validates score range, finite values, both classes and exact vector shape.
    evaluated = evaluate_product(a.activity.to_numpy(int), a.probability.to_numpy(), b.probability.to_numpy())
    frame = pd.DataFrame({"ligand_id": expected_ids, "activity": a.activity.to_numpy(int),
                          "structure_probability": a.probability.to_numpy(), "rte_probability": b.probability.to_numpy()})
    frame["product_score"] = frame.structure_probability * frame.rte_probability
    return frame, evaluated


def _fit_fold(source, source_run, fit, destination, *, iterations):
    spec = fit["split"]
    train_ids = spec["train_ids"] + spec["validation_ids"]
    test_ids = spec["test_ids"]
    labeled_ids = set(source.loc[source.activity.notna(), "ligand_id"])
    if (len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)) or
            set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != labeled_ids):
        raise ValueError("Structural train/validation/test IDs do not partition all labeled records")
    indexed = source.set_index("ligand_id", drop=False)
    query_ids = source.loc[source.activity.isna(), "ligand_id"].tolist()
    y_train = indexed.loc[train_ids, "activity"].to_numpy(int)
    blocks = {"train": [], "test": [], "query": []}
    selections, feature_names, selection_audits = [], [], []
    for target in TARGETS:
        matrix = residue_matrix(target)
        selector = fit_residue_selector(matrix.loc[train_ids].to_numpy(float), y_train, list(matrix),
                                        inner_splits=5, inner_test_size=.2, seed=42)
        names = [selector.feature_names[i] for i in selector.selected_indices]
        for kind, ids in [("train", train_ids), ("test", test_ids), ("query", query_ids)]:
            blocks[kind].append(selector.transform(matrix.loc[ids].to_numpy(float), list(matrix)))
        feature_names.extend(target + ":" + name for name in names)
        selections.append({"target": target, **selector.diagnostics, "selected_features": names})
        selection_audits.append({"target": target, "ranking": table(selector.ranking, "Train-only ranking").rows,
                                 "inner_scan": table(selector.scan, "Training-only selection diagnostic").rows})
    X = {name: np.column_stack(parts) for name, parts in blocks.items()}
    model = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=.05, loss_function="Logloss",
                               random_seed=42, thread_count=2, verbose=False, allow_writing_files=False,
                               use_best_model=False, task_type="CPU")
    model.fit(X["train"], y_train)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_model(str(destination / "catboost.json"), format="json")
    restored = CatBoostClassifier()
    restored.load_model(str(destination / "catboost.json"), format="json")
    probabilities = model.predict_proba(X["test"])[:, 1]
    delta = float(np.max(np.abs(probabilities - restored.predict_proba(X["test"])[:, 1])))
    if delta > 1e-12:
        raise ValueError("Exported RTE model does not reproduce held-out scores")
    rte = pd.DataFrame({"ligand_id": test_ids, "activity": indexed.loc[test_ids, "activity"].to_numpy(int),
                        "probability": probabilities})
    structural = pd.read_csv(source_run / f"fold_{spec['fold']:02d}" / "heldout_predictions.csv")
    scores, measures = aligned_scores(structural, rte, expected_ids=test_ids)
    scores["fold"] = spec["fold"]
    scores.to_csv(destination / "heldout_predictions.csv", index=False)
    q = pd.read_csv(source_run / f"fold_{spec['fold']:02d}" / "query_predictions.csv")
    if q.ligand_id.duplicated().any() or set(q.ligand_id) != set(query_ids):
        raise ValueError("Structural query scores do not cover all source query IDs")
    q = q.set_index("ligand_id").loc[query_ids].reset_index()
    q = q.rename(columns={"probability": "structure_probability"})
    q["rte_probability"] = restored.predict_proba(X["query"])[:, 1]
    if not np.isfinite(q[["structure_probability", "rte_probability"]]).all().all():
        raise ValueError("Nonfinite query scores")
    q["product_score"] = q.structure_probability * q.rte_probability
    q.to_csv(destination / "query_predictions.csv", index=False)
    write_json(destination / "feature_names.json", feature_names)
    write_json(destination / "selection_audit.json", selection_audits)
    write_json(destination / "partitions.json", {"rte_train_ids": train_ids, "test_ids": test_ids,
                                                "dmpnn_partitions": spec})
    values = {"fold": spec["fold"], "n_rte_train": len(train_ids), "n_test": len(test_ids),
              "metrics": measures, "model_reload_max_error": delta, "selections": selections,
              "dmpnn_fold_sha256": digest(source_run / f"fold_{spec['fold']:02d}" / "result.json")}
    values["artifacts"] = {path.name: digest(path) for path in destination.iterdir() if path.is_file() and path.name != "result.json"}
    write_json(destination / "result.json", values)
    return values


def fusion_runs(protocol="paired_cv", source_split_indices=None, max_epochs=60, iterations=500):
    if not isinstance(iterations, int) or not 10 <= iterations <= 1000:
        raise ValueError("Declare10..1000CatBoost iterations")
    structural = validate_structure(protocol, source_split_indices, max_epochs, False)
    source = load()
    config = {"iterations": iterations, "depth": 6, "learning_rate": .05, "seed": 42,
              "rte_training": "complete outer training partition including DMPNN early-stopping validation",
              "selection": "AppendixA,refitted per outer training partition,inner5randomholdouts,98%energy",
              "threshold": .5}
    package = Path(__file__).resolve().parents[1]
    code = {"fusion.py": digest(Path(__file__)), "common/rmt.py": digest(package / "common/rmt.py"),
            "common/classification.py": digest(package / "common/classification.py")}
    software = {name: version(name) for name in ["catboost", "numpy", "pandas", "scikit-learn", "rdkit"]}
    structural_root = Path(os.getenv("CHEM_REPRO_STRUCTURE_RESULTS_DIR", "structure_validation_runs")).resolve()
    output_root = Path(os.getenv("CHEM_REPRO_FUSION_RESULTS_DIR", "fusion_validation_runs")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for description in structural.metadata.protocol_parameters["worker_runs"]:
        cache_id = description["cache_id"]
        if Path(cache_id).name != cache_id:
            raise ValueError("Invalid structural cache identifier")
        source_run = structural_root / cache_id
        saved = json.loads((source_run / "result.json").read_text())
        if saved["run_spec"] != description["run_spec"]:
            raise ValueError("Structural cache changed after validation")
        request = {"protocol": "cannabis.paired_product.v1", "config": config, "code": code,
                   "software": software, "source_manifest": verified_manifest("cannabis"),
                   "dmpnn_result_sha256": digest(source_run / "result.json")}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        directory = output_root / (description["scheme"] + "_" + key[:16])
        directory.mkdir(exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This exact paired fusion is already running") from exc
            request_path = directory / "request.json"
            if request_path.exists() and json.loads(request_path.read_text()) != request:
                raise ValueError("Fusion cache request differs")
            write_json(request_path, request)
            folds, heldout, query = [], [], []
            for path in sorted(source_run.glob("fold_*/result.json")):
                fit = json.loads(path.read_text())
                destination = directory / path.parent.name
                result_path = destination / "result.json"
                if result_path.exists():
                    completed = json.loads(result_path.read_text())
                    if completed["dmpnn_fold_sha256"] != digest(path):
                        raise ValueError("Cached RTE branch belongs to a different structural fold")
                    for name, checksum in completed["artifacts"].items():
                        if digest(destination / name) != checksum:
                            raise ValueError("Fusion artifact checksum mismatch")
                else:
                    completed = _fit_fold(source, source_run, fit, destination, iterations=iterations)
                folds.append(completed)
                heldout.append(pd.read_csv(destination / "heldout_predictions.csv"))
                query.append(pd.read_csv(destination / "query_predictions.csv"))
            h, q = pd.concat(heldout, ignore_index=True), pd.concat(query, ignore_index=True)
            if description["scheme"] != "source_random" and (h.ligand_id.duplicated().any() or
                    set(h.ligand_id) != set(source.loc[source.activity.notna(), "ligand_id"])):
                raise ValueError("Paired OOF predictions do not cover the labeled source exactly once")
            h.to_csv(directory / "heldout_predictions.csv", index=False)
            q.to_csv(directory / "query_predictions_by_fold.csv", index=False)
            average = q.groupby("ligand_id", as_index=False).agg(
                structure_probability=("structure_probability", "mean"), rte_probability=("rte_probability", "mean"),
                mean_fold_product=("product_score", "mean"), n_models=("fold", "nunique"))
            average["product_score"] = average.structure_probability * average.rte_probability
            average = source.loc[source.activity.isna(), ["ligand_id", "synID", "SMILES"]].merge(
                average, on="ligand_id", how="left", validate="one_to_one")
            if (average.n_models != len(folds)).any():
                raise ValueError("Query ensemble lacks a completed fold")
            average.to_csv(directory / "query_mean_scores.csv", index=False)
            measures = evaluate_product(h.activity.to_numpy(int), h.structure_probability, h.rte_probability) if description["scheme"] != "source_random" else None
            final = {"scheme": description["scheme"], "n_fits": len(folds), "config": config,
                     "pooled_oof_metrics": measures, "folds": folds, "request_sha256": digest(request_path)}
            write_json(directory / "result.json", final)
            rows.append((directory, final))
    return structural, rows


def evaluate_fusion(protocol="paired_cv", source_split_indices=None, max_epochs=60, iterations=500, include_models=False):
    structural, runs = fusion_runs(protocol, source_split_indices, max_epochs, iterations)
    metrics, effects, artifacts = [], [], []
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    for directory, info in runs:
        for fold in info["folds"]:
            for branch in ["structure", "rte", "product"]:
                metrics.append({"scheme": info["scheme"], "scope": "fold", "fold": fold["fold"], "branch": branch, **fold["metrics"][branch]})
            effects.append({"scheme": info["scheme"], "scope": "fold", "fold": fold["fold"], **fold["metrics"]["operating_point_effect"]})
        if info["pooled_oof_metrics"]:
            for branch in ["structure", "rte", "product"]:
                metrics.append({"scheme": info["scheme"], "scope": "pooled_oof", "fold": None, "branch": branch, **info["pooled_oof_metrics"][branch]})
            effects.append({"scheme": info["scheme"], "scope": "pooled_oof", "fold": None, **info["pooled_oof_metrics"]["operating_point_effect"]})
        for name in ["result.json", "request.json", "heldout_predictions.csv", "query_predictions_by_fold.csv", "query_mean_scores.csv"]:
            artifacts.append(save_artifact((directory / name).read_bytes(), suffix=name.rsplit(".", 1)[1],
                             media_type="application/json" if name.endswith("json") else "text/csv", public_base_url=base,
                             display_name=info["scheme"] + "_fusion_" + name))
        if include_models:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(directory.glob("fold_*/*")):
                    if path.is_file():
                        archive.write(path, str(path.relative_to(directory)))
                archive.write(directory / "request.json", "request.json")
            artifacts.append(save_artifact(buffer.getvalue(), suffix="zip", media_type="application/zip",
                                           public_base_url=base, display_name=info["scheme"] + "_rte_fusion_models.zip"))
    output = result("Compared actual DMPNN and RTE/CatBoost predictions and their product on exactly matched held-out IDs. "
                    "At threshold0.5 the product can only remove positive predictions from either branch; a lower FPR "
                    "must be considered alongside recall loss and ranking metrics. The product is not a calibrated joint posterior.",
                    [metric("rte_fits_on_structural_partitions", sum(info["n_fits"] for _, info in runs))],
                    {"paired_metrics": table(pd.DataFrame(metrics), "Matched held-out observations; fold and pooled OOF scopes separate"),
                     "operating_point_effects": table(pd.DataFrame(effects), "Product minus structural branch; positive retention is not dataset coverage")},
                    "cannabis.paired_product.v1", notes=[
                        EvidenceNote(code="FUSION_RECONSTRUCTION", message="Reconstruction on3171available labeled source records. "
                                     "Author final fold assignments/checkpoints are unavailable; the paper12.20%to4.92%FPR is a reported "
                                     "reference, not an acceptance threshold for these computations. Both branches use the same outer "
                                     "test IDs. RTE training includes the DMPNN validation subset, never outer-test rows. Inner RTE "
                                     "feature-selection validation is random even for scaffold outer folds; no outer scaffold leakage is introduced."),
                        EvidenceNote(code="QUERY_ENSEMBLE_CHOICE", message="For querying2749source records, component probabilities "
                                     "are averaged over completed folds, then multiplied. Mean of per-fold products is exported separately. "
                                     "No final full-cohort refit or author ensemble-equivalence is claimed.")],
                    protocol_choice=protocol, max_epochs=max_epochs, iterations=iterations,
                    fold_runs=[{"scheme":info["scheme"], "request_sha256":info["request_sha256"]} for _,info in runs])
    output.status = "partial"
    output.answer.limitations.extend(structural.answer.limitations)
    output.metadata.sources.extend(s for s in structural.metadata.sources if s.name not in {v.name for v in output.metadata.sources})
    output.artifacts.extend(artifacts)
    return export_tables(output, port=7341)
