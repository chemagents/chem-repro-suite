"""Verify and aggregate completed RTE fits without representing cached training as a new fit."""

import hashlib
import io
import json
import os
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.artifacts import artifact_root, export_tables, save_artifact
from ..common.classification import binary_metrics
from ..common.contracts import AnalysisResult, EvidenceNote
from ..common.datasets import table, verified_manifest
from .analysis import metric, result
from .data import TARGETS, split
from .validation import validate_rte

ALGORITHM_FILES = ["cannabis/validation.py", "cannabis/data.py", "common/rmt.py",
                   "common/classification.py", "common/chemistry.py"]


def _verified_archive(directory):
    archive = directory / "executed_code.zip"
    package = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(name for name in zipped.namelist() if name.startswith("chem_repro/") and name.endswith(".py"))
        if len(names) != len(set(names)):
            raise ValueError("Execution archive contains duplicate source members")
        digest = hashlib.sha256()
        for name in names:
            digest.update(name.removeprefix("chem_repro/").encode() + b"\0")
            digest.update(zipped.read(name))
        for name in ALGORITHM_FILES:
            if zipped.read("chem_repro/" + name) != (package / name).read_bytes():
                raise ValueError("RTE scientific implementation changed; use a new verified run directory")
    return archive, digest.hexdigest()


def _read_predictions(cached, *, index, iterations, implementation, artifact_directory):
    if cached.metadata.implementation_sha256 != implementation:
        raise ValueError("Cached RTE result does not match its executed-code archive")
    parameters = cached.metadata.protocol_parameters
    if parameters["split_indices"] != [index] or parameters["catboost"]["iterations"] != iterations:
        raise ValueError("Cached split or training configuration differs from the request")
    for name in ["rdkit", "numpy", "pandas", "scipy", "scikit-learn", "catboost"]:
        if cached.metadata.software.get(name) != version(name):
            raise ValueError("RTE cache scientific software differs: " + name)
    expected_sources = verified_manifest("cannabis")["files"]
    if {s.name: s.sha256 for s in cached.metadata.sources} != {n:v["sha256"] for n,v in expected_sources.items()}:
        raise ValueError("RTE cache source data differ")
    payloads = {}
    for artifact in cached.artifacts:
        # Read from the configured local cache only; no paths or credentials from cached URLs.
        suffix = artifact.name.rsplit(".", 1)[-1]
        path = artifact_directory / (artifact.sha256 + "." + suffix)
        body = path.read_bytes()
        if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
            raise ValueError("RTE cache artifact digest/size mismatch")
        payloads[artifact.name] = body
    payload = json.loads(payloads["cannabis_rte_validation_predictions.json"])
    if payload["metadata"] != cached.metadata.model_dump(mode="json"):
        raise ValueError("Prediction artifact and cached result provenance disagree")
    predictions = pd.DataFrame(payload["predictions"])
    if set(predictions.split_index) != {index} or predictions.duplicated(["target", "model", "ligand_id"]).any():
        raise ValueError("RTE predictions contain mixed splits or duplicate molecule scores")
    expected_groups = {(target, model) for target in [*TARGETS, "ALL6"] for model in ["CatBoost", "LinearSVM"]}
    if set(predictions.groupby(["target", "model"]).groups) != expected_groups:
        raise ValueError("RTE cache lacks target/model combinations")
    train, test = split(index)
    labels = test.set_index("ligand_id").activity.astype(int)
    saved_metrics = {(row["target"], row["model"]):row for row in cached.answer.tables["heldout_metrics"].rows}
    computed = []
    for (target, model), frame in predictions.groupby(["target", "model"]):
        if set(frame.ligand_id) != set(test.ligand_id):
            raise ValueError("Cached test scores do not cover the declared source holdout")
        frame = frame.set_index("ligand_id").loc[test.ligand_id]
        if not np.array_equal(frame.label.to_numpy(int), labels.loc[test.ligand_id].to_numpy(int)):
            raise ValueError("Cached test labels differ from source data")
        kind, threshold = ("probability", .5) if model == "CatBoost" else ("decision_margin", 0.)
        if set(frame.score_kind) != {kind}:
            raise ValueError("SVM margins must not be silently treated as probabilities")
        checked = binary_metrics(frame.label.to_numpy(int), frame.score.to_numpy(float), threshold=threshold, score_kind=kind)
        stored = saved_metrics[(target, model)]
        for name, value in checked.items():
            other = stored[name]
            if isinstance(value, float):
                if not isinstance(other, (int, float)) or not np.isclose(value, other, rtol=0, atol=1e-12):
                    raise ValueError("Cached metric disagrees with saved held-out scores: " + name)
            elif other != value:
                raise ValueError("Cached count or score definition differs: " + name)
        computed.append({"split_index": index, "target":target, "model":model,
                         "n_selected_features":stored["n_selected_features"], **checked})
    for selection in payload["selections_and_splits"]:
        registry = selection["split_registry"]
        if (selection["split_index"] != index or registry["train_ids"] != train.ligand_id.astype(int).tolist()
                or registry["test_ids"] != test.ligand_id.astype(int).tolist()
                or registry["train_labels"] != train.activity.astype(int).tolist()
                or registry["test_labels"] != test.activity.astype(int).tolist()):
            raise ValueError("Residue-selection training population differs from the source split")
    if {s["target"] for s in payload["selections_and_splits"]} != set(TARGETS):
        raise ValueError("Residue-selection evidence does not cover all six targets")
    return computed, payloads, predictions


def validate_rte_cached(split_indices=None, *, iterations=500, include_models=False):
    indices = list(range(30)) if split_indices is None else split_indices
    if (not indices or len(set(indices)) != len(indices) or
            any(not isinstance(i, int) or not 0 <= i < 30 for i in indices)):
        raise ValueError("Require unique source split indices in0..29")
    configured = os.getenv("CHEM_REPRO_RTE_RESULTS_DIR")
    if not configured:
        return export_tables(validate_rte(indices, iterations=iterations, include_models=include_models), port=7341)
    directory = Path(configured).resolve()
    first = json.loads((directory / f"split_{indices[0]:02d}.json").read_text())
    if first["metadata"]["protocol_parameters"]["catboost"]["iterations"] != iterations:
        # A user-selected training budget is a different experiment, not a cache corruption.
        # Fit that exact request without replacing the historical execution directory.
        fresh = validate_rte(indices, iterations=iterations, include_models=include_models)
        fresh.answer.limitations.append(EvidenceNote(
            code="NEW_REQUESTED_TRAINING_BUDGET", severity="info",
            message="The requested iteration count differs from the configured historical cache. "
                    "Fresh models were fitted using the requested count; historical fits were preserved.",
        ))
        return export_tables(fresh, port=7341)
    archive, implementation = _verified_archive(directory)
    cached_artifacts = Path(os.getenv("CHEM_REPRO_RTE_ARTIFACTS_DIR", str(artifact_root()))).resolve()
    frames, selections, audits, artifacts, execution_records = [], [], [], [], []
    prediction_csv = io.StringIO()
    audit_zip = io.BytesIO()
    archive_writer = zipfile.ZipFile(audit_zip, "w", zipfile.ZIP_DEFLATED)
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    for index in indices:
        source = directory / f"split_{index:02d}.json"
        cached = AnalysisResult.model_validate_json(source.read_bytes())
        rows, payloads, predictions = _read_predictions(cached, index=index, iterations=iterations,
                                                        implementation=implementation, artifact_directory=cached_artifacts)
        predictions.to_csv(prediction_csv, index=False, header=(index == indices[0]))
        frames.extend(rows)
        selections.extend(cached.answer.tables["residue_selections"].rows)
        audits.extend(cached.answer.tables["split_audit"].rows)
        execution_records.append({"split_index":index, "result_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                                  "implementation_sha256":implementation, "software":cached.metadata.software})
        for original in cached.artifacts:
            if original.name.endswith("models.zip"):
                if include_models:
                    artifacts.append(save_artifact(payloads[original.name], suffix="zip",
                                     media_type=original.media_type, public_base_url=base,
                                     display_name=f"split_{index:02d}_" + original.name))
            else:
                archive_writer.writestr(f"split_{index:02d}/" + original.name, payloads[original.name])
        if include_models and not any(name.endswith("models.zip") for name in payloads):
            raise ValueError("Requested fitted models are missing from the verified cache")
    archive_writer.close()
    artifacts.extend([
        save_artifact(prediction_csv.getvalue().encode(), suffix="csv", media_type="text/csv",
                      public_base_url=base, display_name="rte_all_heldout_predictions.csv"),
        save_artifact(audit_zip.getvalue(), suffix="zip", media_type="application/zip",
                      public_base_url=base, display_name="rte_original_predictions_and_selection_audits.zip"),
    ])
    frame = pd.DataFrame(frames)
    summary = frame.groupby(["target", "model"]).agg(
        n_splits=("split_index", "nunique"), mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"),
        mean_average_precision=("average_precision", "mean"), std_average_precision=("average_precision", "std"),
        mean_fpr=("fpr", "mean"), std_fpr=("fpr", "std"), mean_recall=("recall", "mean"), std_recall=("recall", "std"),
    ).reset_index()
    output = result("Verified previously executed RTE fits against the exact archived algorithm, scientific software, "
                    "source records, split memberships and complete held-out scores. Metrics were independently recomputed "
                    "from those scores; no training was repeated and no paper-reported metric was substituted.",
                    [metric("completed_model_fits",len(frame)), metric("completed_outer_splits",len(indices))],
                    {"heldout_metrics":table(frame,"Recomputed from verified cached predictions"),
                     "split_summary":table(summary,"Mean and sample SD across repeated source holdouts; no pooled OOF claim"),
                     "residue_selections":table(pd.DataFrame(selections),"Archived train-only selection for each target/split"),
                     "split_audit":table(pd.DataFrame(audits),"Verified exact source populations")},
                    "cannabis.rte_cached_validation.v1", notes=[*cached.answer.limitations,
                        EvidenceNote(code="VERIFIED_CACHED_TRAINING",severity="info",message="These are actual earlier CPU fits. "
                                     "The current request verifies and aggregates them. Execution source/software and result hashes "
                                     "are preserved; current wrapper provenance is separate from original model execution.")],
                    split_indices=indices, iterations=iterations, cached_execution=execution_records,
                    archived_implementation_sha256=implementation)
    output.status = "partial"
    output.metadata.reused_from_cache = True
    artifacts.append(save_artifact(archive.read_bytes(),suffix="zip",media_type="application/zip",
                                  public_base_url=base,display_name="rte_executed_code.zip"))
    output.artifacts.extend(artifacts)
    return export_tables(output,port=7341)
