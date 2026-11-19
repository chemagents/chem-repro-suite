"""Expose completed, independently replayed nested-v2 runs through the existing tool."""

import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from ..common.artifacts import save_artifact, save_json, save_table
from ..common.contracts import AnalysisResult, Answer, DataSource, EvidenceNote, Metric, Provenance
from ..common.datasets import DATA_ROOT, implementation_sha256, provenance, table, verified_manifest
from ..common.worker_environment import worker_environment
from .candidate_identity import attach_mapping_provenance, identity_map, toxicity_rows, toxicity_tables
from .paper2024 import xgboost_search_space

ROOT = Path(__file__).resolve().parents[1]


def checked(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Completed nested run changed after verification")
    return data


def zip_member(archive, name, content):
    # Stable container metadata lets identical replay requests reuse one artifact
    # instead of storing another timestamp-only copy of every fitted model.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content)


def search_audit(root, result):
    """Requested trials are not successful fits; preserve backend failures explicitly."""
    config = result["config"]
    rows, failures = [], []
    grid_n = len(xgboost_search_space(full=config["endpoint"] in {
        "mouse_oral_ld50", "mouse_intraperitoneal_ld50", "mouse_intravenous_ld50", "bbb"}))
    for scheme in config["schemes"]:
        for fold in range(config["outer_folds"]):
            prefix = f"{scheme}_fold{fold}/"

            def read(name, directory=prefix):
                path = directory + name
                return json.loads(checked(root / path, result["artifacts"][path]))

            trials = read("optuna_trials.json")
            if len(trials) != config["optuna_trials"] or any(t["state"] not in {"COMPLETE", "FAIL", "PRUNED"} for t in trials):
                raise ValueError("Incomplete Optuna trial history in a completed run")
            for family in ["CatBoost_ECFP", "XGBoost_fragments"]:
                completed = read(family + "_completed_candidates.json")
                if family == "CatBoost_ECFP":
                    done = {t["number"] for t in trials if t["state"] == "COMPLETE"}
                    if {t["number"] for t in completed} != done:
                        raise ValueError("Completed candidate scores disagree with trial states")
                    failed = [t for t in trials if t["state"] == "FAIL"]
                    pruned = sum(t["state"] == "PRUNED" for t in trials)
                    failures.extend({"endpoint": config["endpoint"], "scheme": scheme, "fold": fold,
                                     "candidate": t["number"], "parameters_json": json.dumps(t["parameters"], sort_keys=True),
                                     "backend_error": t["user_attrs"].get("backend_failure", "unspecified")}
                                    for t in failed)
                    requested = config["optuna_trials"]
                else:
                    if len(completed) != grid_n or {t["number"] for t in completed} != set(range(grid_n)):
                        raise ValueError("Published finite XGBoost grid was not fully evaluated")
                    requested, failed, pruned = grid_n, [], 0
                rows.append({"endpoint": config["endpoint"], "scheme": scheme, "fold": fold, "model": family,
                             "requested_candidates": requested, "completed_candidates": len(completed),
                             "failed_candidates": len(failed), "pruned_candidates": pruned,
                             "successful_candidate_inner_fold_evaluations": sum(len(t["inner_fold_metrics"]) for t in completed),
                             "exhaustive_finite_grid": family == "XGBoost_fragments"})
    return rows, failures


def _evaluate_available(endpoints=None):
    source_manifest = verified_manifest("open_syntelly_2024")
    data_root = DATA_ROOT / "open_syntelly_2024"
    protocol = json.loads((data_root / "protocol.json").read_text())
    endpoints = list(protocol["files"]) if endpoints is None else endpoints
    if not endpoints or len(set(endpoints)) != len(endpoints) or set(endpoints) - set(protocol["files"]):
        raise ValueError("Require unique known endpoints; none are silently omitted")
    registry_path = os.environ.get("CHEM_REPRO_NESTED_TOX_RUNS_FILE")
    if not registry_path:
        raise RuntimeError("Configure CHEM_REPRO_NESTED_TOX_RUNS_FILE with completed nested-v2 runs")
    registry = json.loads(Path(registry_path).read_text())
    missing = sorted(set(endpoints) - set(registry))
    if missing:
        raise RuntimeError("Completed nested-v2 models unavailable for: " + ", ".join(missing))
    python = os.environ.get("CHEM_REPRO_TOX_PYTHON")
    addon = os.environ.get("CHEM_REPRO_TOX_TUNING_ADDON")
    if not python or not addon:
        raise RuntimeError("Configure the pinned toxicity Python and Optuna addon before verification")
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    comparisons, validations, preparation, folds, metrics, sources, artifacts = [], [], [], [], [], [], []
    search_rows, failed_candidates = [], []
    candidates = {"identity_random": [], "scaffold": []}
    runs, software = {}, None
    mapping = identity_map()
    for endpoint in endpoints:
        root = Path(registry[endpoint]["result_dir"]).resolve()
        source = Path(registry[endpoint]["source_dir"]).resolve()
        result_bytes = (root / "result.json").read_bytes()
        result_hash = hashlib.sha256(result_bytes).hexdigest()
        result = json.loads(result_bytes)
        spec, config = protocol["files"][endpoint], result["config"]
        if config["endpoint"] != endpoint or config["training_sha256"] != spec["sha256"]:
            raise ValueError("Nested run is not fitted on the declared endpoint source")
        if config["query_sha256"] != source_manifest["files"]["cannabis_query.csv"]["sha256"]:
            raise ValueError("Nested run query is not the fixed Cannabis source cohort")
        if config["task"] != spec["task"] or config["unit"] != spec["unit"]:
            raise ValueError("Nested run task or target unit differs from source declaration")
        with tempfile.TemporaryDirectory(prefix="nested-tox-verify-") as tmp, worker_environment(source / "chem_repro", extra_import_roots=[addon]) as environment:
            report = Path(tmp) / "verification.json"
            completed = subprocess.run([python, str(ROOT / "runtime/nested_toxicity_verify.py"),
                "--result-dir", str(root), "--source-dir", str(source), "--report", str(report)],
                env=environment,
                capture_output=True, text=True, timeout=900, check=False)
            if completed.returncode:
                raise RuntimeError("Nested model replay failed: " + completed.stderr[-2000:])
            verification = json.loads(report.read_text())
        if verification["result_sha256"] != result_hash:
            raise ValueError("Run changed during verification")
        audit, failed = search_audit(root, result)
        search_rows.extend(audit)
        failed_candidates.extend(failed)
        if software is not None and software != result["software"]:
            raise ValueError("Requested nested runs use inconsistent scientific environments")
        software = result["software"]
        runs[endpoint] = {"result_sha256": result_hash, "outer_folds": config["outer_folds"],
                          "inner_folds": config["inner_folds"], "seed": config["seed"],
                          "optuna_trials": config["optuna_trials"], "optuna_startup_trials": config["optuna_startup_trials"],
                          "max_xgboost_candidates": config["max_xgboost_candidates"],
                          "assumptions": result["assumptions"],
                          "fitted_code_sha256": json.loads((root / "code_hashes.json").read_text()),
                          "verifier_sha256": verification["verifier_sha256"]}
        preparation.append({"endpoint": endpoint, **result["preparation"], "applied_to_predictions": True})
        metrics.append(Metric(name=endpoint + "_prepared_training_molecules", value=result["preparation"]["aggregated_molecules"],
                              population="Prepared public training source"))
        folds.extend({"endpoint": endpoint, **row} for row in verification["fold_audit"])
        for scheme, models in verification["validation"].items():
            for family, values in models.items():
                validations.extend({"endpoint": endpoint, "scheme": scheme, "model": family,
                                    "metric": name, "value": value} for name, value in values.items())
            filename = f"{scheme}_query_predictions.csv"
            content = checked(root / filename, result["artifacts"][filename])
            predictions = pd.read_csv(io.BytesIO(content), dtype={"query_id": str})
            candidates[scheme].extend(toxicity_rows(mapping, predictions, endpoint, spec))
            for cohort, rows in predictions.groupby("cohort", sort=True):
                comparisons.append({"endpoint": endpoint, "scheme": scheme, "cohort": cohort,
                    "n": len(rows), "unit": spec["unit"], "median_prediction": float(rows.prediction.median()),
                    "median_domain_score_percent": float(rows.reference_domain_score_percent.median()),
                    "outside_reference_distance_fraction": float(rows.outside_reference_distance.mean()),
                    "training_identity_overlap_n": int(rows.training_identity_overlap.sum()),
                    "training_scaffold_overlap_n": int(rows.training_scaffold_overlap.sum()),
                    "outside_training_element_policy_n": int((~rows.matches_training_element_policy).sum())})
            artifacts.append(save_artifact(content, suffix="csv", media_type="text/csv", public_base_url=base,
                                           display_name=f"{endpoint}_{filename}"))
        # Preserve models, trial records, exact inputs and fitted source code as a
        # single downloadable archive. No fitted state is synthesized from prose.
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, expected in result["artifacts"].items():
                path = (root / name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Artifact path escapes the run")
                zip_member(archive, "results/" + name, checked(path, expected))
            zip_member(archive, "results/result.json", checked(root / "result.json", result_hash))
            code = json.loads((root / "code_hashes.json").read_text())
            for name, expected in code.items():
                fitted_package = (source / "chem_repro").resolve()
                code_path = (fitted_package / name).resolve()
                if not code_path.is_relative_to(fitted_package):
                    raise ValueError("Fitted source path escapes the package")
                zip_member(archive, "source/chem_repro/" + name, checked(code_path, expected))
            replay_config = dict(config)
            for key in ["training", "query"]:
                zip_member(archive, f"source/data/{key}.csv", checked(source / config[key + "_path"], config[key + "_sha256"]))
                replay_config[key + "_path"] = f"data/{key}.csv"
            replay_config["output"] = "replayed_results"
            for key in ["addon_path", "addon_lock_path", "addon_lock_sha256"]:
                replay_config.pop(key, None)
            zip_member(archive, "source/replay_config.json", json.dumps(replay_config, indent=2))
            zip_member(archive, "source/requirements.txt", "".join(f"{name}=={value}\n" for name, value in sorted(software.items())))
            zip_member(archive, "README.txt", "Verified saved models and original run evidence are under results/.\n"
                "For a new fit, use Python3.12, enter source/, install requirements.txt in a fresh environment, then run:\n"
                "python -m chem_repro.runtime.syntelly_nested_worker --config replay_config.json\n"
                "The exported replay config changes only file locations/output directory and removes the local addon bootstrap paths.\n"
                "Scientific settings/input hashes are retained; original config is results/config.json.\n"
                "This is a declared bounded public-data reconstruction, not original author weights or an optimality claim.\n")
            zip_member(archive, "verified_replay.json", json.dumps(verification, indent=2))
        artifacts.append(save_artifact(bundle.getvalue(), suffix="zip", media_type="application/zip",
                                       public_base_url=base, display_name=f"{endpoint}_nested_models_and_evidence.zip"))
        sources.append(DataSource(name=endpoint, sha256=spec["sha256"], source_url=spec["source"],
                                  revision="public-source-nested-method-v2", derivation=spec["notes"],
                                  parent_sha256=spec.get("parent_sha256")))
    for name in ["protocol.json", "cannabis_query.csv", "paper2024.xml", "paper2024_supplement.pdf"]:
        item = source_manifest["files"][name]
        sources.append(DataSource(name=name, sha256=item["sha256"], source_url=item["source_url"],
                                  revision=source_manifest["revision"]))
    tables = {"cohort_comparisons": table(pd.DataFrame(comparisons), "New nested model predictions, separately by split scheme"),
              "model_validation": table(pd.DataFrame(validations), "Pooled outer OOF metrics after inner-only tuning/selection"),
              "training_preparation": table(pd.DataFrame(preparation), "2024 preparation applied to these new model fits"),
              "nested_fold_audit": table(pd.DataFrame(folds), "Verified inner/outer grouping and no outer test rows in search"),
              "tuning_search_audit": table(pd.DataFrame(search_rows), "Attempted candidates, successful candidate metrics and failures are distinct"),
              "failed_search_candidates": table(pd.DataFrame(failed_candidates, columns=[
                  "endpoint", "scheme", "fold", "candidate", "parameters_json", "backend_error"]),
                  "Retained failed candidates; partial inner fits within a failed candidate are not fully counted")}
    for scheme, records in candidates.items():
        frame, named = toxicity_tables(records)
        tables.update({scheme + "_" + name: value for name, value in named.items()})
        artifacts.append(save_table(table(frame, "All ten mapped candidate values for this nested split scheme"),
                                    port=7341, display_name=scheme + "_nested_candidate_predictions.csv"))
    result = AnalysisResult(status="partial", answer=Answer(
        finding=f"Completed nested public-data reconstruction for {len(endpoints)} explicitly requested endpoints: "
                "five outer identity-random/scaffold folds, inner-only tuning and fragment selection, verified saved-model "
                "predictions for all source compounds and the ten mapped Figure4 candidates. These are new fitted models; "
                "the original Syntelly outputs and historical optimization are not recovered.",
        metrics=metrics, tables=tables, limitations=[
            EvidenceNote(code="DECLARED_2024_RECONSTRUCTION", message="RDKit fragments replace the unavailable author dictionary. "
                         "Count-filter denominator, correlation ordering, original source labels and historical tuning choices "
                         "are not fully specified; every run records assumptions and its finite Optuna budget. No convergence claim."),
            EvidenceNote(code="CHANGED_TRAINING_COHORT", message="Preparation is applied to v2 fits. Differences from fixed-v1 "
                         "metrics combine cohort changes, feature selection, tuning and validation changes; do not attribute "
                         "the difference to one algorithmic step or compare it as a paired improvement."),
            EvidenceNote(code="SEARCH_FAILURE_ACCOUNTING", message=f"{len(failed_candidates)} search candidates failed. "
                         "Failed candidates retain their parameters and backend errors; they are not counted as successful "
                         "inner evaluations or silently retried. Partial fits within failed candidates are not completely "
                         "counted. All selected outer models are independently replayed. Optuna is a finite declared search, "
                         "not an exhaustive or converged optimum claim."),
            EvidenceNote(code="PUBLIC_DATA_AND_DOMAIN_SCOPE", message="Source query cohorts and their sizes differ from the final "
                         "paper cohorts. kNN domain scores are relative, not calibrated accuracy. Training overlap and queries "
                         "outside the training element policy are explicit. DILI is not a separate broad hepatotoxicity output; "
                         "hERG is channel blockade, aquatic units remain the declared source scale.")]),
        metadata=Provenance(study_id="cannabis_2026", paper_doi="10.3390/ijms27188000",
                            protocol_id="paper2024_public_nested_v2", protocol_parameters={"runs": runs},
                            sources=sources, software=software, implementation_sha256=implementation_sha256(),
                            reused_from_cache=True), artifacts=artifacts)
    attach_mapping_provenance(result, base_url=base)
    return result


def evaluate(endpoints=None):
    """Replay available requested endpoints and explicitly retain missing coverage.

    An absent completed model cannot erase other independently verified endpoints.
    Invalid IDs, corrupt runs and provenance failures remain errors; no recipe is
    substituted and missing predictions are never generated from reference values.
    """
    verified_manifest("open_syntelly_2024")
    protocol = json.loads((DATA_ROOT / "open_syntelly_2024/protocol.json").read_text())
    requested = list(protocol["files"]) if endpoints is None else list(endpoints)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(protocol["files"]):
        raise ValueError("Require unique known endpoints; none are silently omitted")
    registry_path = os.environ.get("CHEM_REPRO_NESTED_TOX_RUNS_FILE")
    if not registry_path:
        raise RuntimeError("Configure CHEM_REPRO_NESTED_TOX_RUNS_FILE with completed nested-v2 runs")
    registry = json.loads(Path(registry_path).read_text())
    available = [e for e in requested if e in registry]
    missing = [e for e in requested if e not in registry]
    if available:
        output = _evaluate_available(available)
    else:
        output = AnalysisResult(status="unsupported", answer=Answer(
            finding="None of the requested endpoints has a completed nested-v2 run. No predictions returned.",
            metrics=[], tables={}, limitations=[]),
            metadata=provenance("open_syntelly_2024", "paper2024_public_nested_v2"))
    output.answer.tables["requested_endpoint_coverage"] = table(pd.DataFrame([
        {"endpoint": e, "requested": True,
         "status": "verified_completed_run" if e in available else "unavailable_completed_run",
         "recipe": "nested_paper2024_v2", "predictions_returned": e in available}
        for e in requested]), "Every requested endpoint, including unavailable models; no recipe substitution")
    output.metadata.protocol_parameters["requested_endpoints"] = requested
    output.metadata.protocol_parameters["completed_endpoints"] = available
    output.metadata.protocol_parameters["unavailable_endpoints"] = missing
    if missing:
        output.answer.finding += (f" Endpoint coverage: {len(available)}/{len(requested)} requested endpoints; "
                                  "completed nested models unavailable for: " + ", ".join(missing) + ".")
        output.answer.limitations.append(EvidenceNote(code="REQUESTED_ENDPOINTS_UNAVAILABLE",
            message="No completed nested-v2 model for: " + ", ".join(missing) +
                    ". These endpoints have no returned predictions. Other requested endpoints are verified independently; "
                    "fixed-v1 predictions and paper-reported values are not substituted."))
        if available:
            output.status = "partial"
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    output.artifacts.append(save_json(output.model_dump(mode="json"), public_base_url=base,
                                     display_name="nested_toxicity_results.json"))
    return output
