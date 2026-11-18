"""New endpoint predictions from the open 2024 recipe, with source-scoped comparisons."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.artifacts import save_artifact, save_json, save_table
from ..common.contracts import AnalysisResult, Answer, DataSource, EvidenceNote, Metric, Provenance
from ..common.datasets import implementation_sha256, table, verified_manifest
from ..common.worker_environment import worker_environment
from .candidate_identity import attach_mapping_provenance, identity_map, toxicity_rows, toxicity_tables
from .paper2024 import ASSUMPTIONS, prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/open_syntelly_2024"
WORKER = ROOT / "runtime/syntelly_worker.py"
RECIPE = ROOT / "vendor/cannabis_open_tox"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_toxicity_profile(endpoints=None):
    source_manifest = verified_manifest("open_syntelly_2024")
    for name in ["protocol.json", "cannabis_query.csv"]:
        if sha(DATA / name) != source_manifest["files"][name]["sha256"]:
            raise ValueError("Toxicity protocol/query bytes changed after manifest validation")
    manifest = json.loads((DATA / "protocol.json").read_text())
    endpoints = list(manifest["files"]) if endpoints is None else endpoints
    if not endpoints or len(set(endpoints)) != len(endpoints) or set(endpoints) - set(manifest["files"]):
        raise ValueError("Require a nonempty unique list of available endpoints")
    python = os.getenv("CHEM_REPRO_TOX_PYTHON")
    if not python or not Path(python).is_file():
        raise RuntimeError("Set CHEM_REPRO_TOX_PYTHON to the isolated open toxicity worker environment")
    locked = dict(line.split("==") for line in
                  (ROOT / "data/cannabis_open_tox/worker_requirements.txt").read_text().splitlines() if line)
    probe = "import json; from importlib.metadata import version; print(json.dumps({n:version(n) for n in " + repr(list(locked)) + "}))"
    with worker_environment(ROOT) as environment:
        actual = json.loads(subprocess.check_output([python, "-c", probe], text=True, timeout=30, env=environment))
    if actual != locked:
        raise RuntimeError("Toxicity worker environment differs from the pinned scientific dependency lock")
    root = Path(os.getenv("CHEM_REPRO_TOX_PROFILE_RESULTS_DIR", "toxicity_profile_runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    query = DATA / "cannabis_query.csv"
    expected = {"query_sha256": sha(query), "worker_sha256": sha(WORKER),
                "recipe_sha256": sha(RECIPE / "tox_train.py"),
                "domain_sha256": sha(ROOT / "common/applicability.py")}
    summaries, validation, metrics, sources, artifacts = [], [], [], [], []
    preparation_summaries = []
    mapped, candidate_predictions = identity_map(), []
    for name in ["protocol.json", "cannabis_query.csv"]:
        item = source_manifest["files"][name]
        sources.append(DataSource(name=name, sha256=item["sha256"], source_url=item["source_url"],
                                  revision=source_manifest["revision"], derivation=item.get("derivation"),
                                  parent_sha256=item.get("parent_sha256")))
    for name in ["paper2024.xml", "paper2024_supplement.pdf"]:
        item = source_manifest["files"][name]
        sources.append(DataSource(name=name, sha256=item["sha256"], source_url=item["source_url"],
                                  revision="Molecules2024-29-1826", derivation=item.get("derivation")))
    reused = True
    for endpoint in endpoints:
        spec = manifest["files"][endpoint]
        train = DATA / spec["path"]
        if sha(train) != spec["sha256"]:
            raise ValueError("Training source digest changed")
        prepared = prepare(pd.read_csv(train, dtype={"source_id": str}, keep_default_na=False), spec["task"],
                           source_dataset=endpoint + ":public_collection")
        preparation_summaries.append({"endpoint": endpoint, **prepared.summary,
                                      "applied_to_returned_predictions": False})
        for label, frame in [("records", prepared.records), ("aggregated_molecules", prepared.molecules)]:
            artifacts.append(save_table(table(frame, "2024 preparation audit; not applied to the frozen v1 model outputs"),
                                        port=7341, display_name=f"{endpoint}_paper2024_{label}.csv"))
        signature = {**expected, "training_sha256": spec["sha256"], "endpoint": endpoint,
                     "task": spec["task"], "unit": spec["unit"], "cv": "both"}
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        run = root / (endpoint + "_" + fingerprint[:16])
        if not (run / "result.json").exists():
            if run.exists():
                raise RuntimeError(f"An unfinished {endpoint} run exists; inspect its log before restarting")
            reused = False
            with (root / (run.name + ".log")).open("w") as log, worker_environment(ROOT) as environment:
                subprocess.run([python, str(WORKER), "--train", str(train), "--query", str(query),
                                "--recipe-root", str(RECIPE), "--endpoint", endpoint,
                                "--task", spec["task"], "--unit=" + spec["unit"], "--cv", "both",
                                "--output", str(run)], check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, timeout=7200)
        result = json.loads((run / "result.json").read_text())
        if any(actual.get(name) != value for name, value in result["spec"]["software"].items()):
            raise ValueError("Cached toxicity results used different scientific dependencies")
        for name, value in {**expected, "training_sha256": spec["sha256"]}.items():
            if result["spec"].get(name) != value:
                raise ValueError("Cached prediction protocol differs from the requested source/code")
        for name, digest in result["artifacts"].items():
            if sha(run / name) != digest:
                raise ValueError("Cached prediction artifact failed its checksum")
        predictions = pd.read_csv(run / "predictions.csv", dtype={"query_id": str})
        candidate_predictions.extend(toxicity_rows(mapped, predictions, endpoint, spec))
        for cohort, rows in predictions.groupby("cohort", sort=True):
            summaries.append({"endpoint": endpoint, "cohort": cohort, "n": len(rows),
                              "unit": spec["unit"], "median_prediction": float(rows.prediction.median()),
                              "positive_fraction_at_0_5": float((rows.prediction >= .5).mean())
                              if spec["task"] == "classification" else None,
                              "median_domain_score_percent": float(rows.reference_domain_score_percent.median()),
                              "outside_reference_distance_fraction": float(rows.outside_reference_distance.mean()),
                              "training_identity_overlap_n": int(rows.training_identity_overlap.sum()),
                              "training_scaffold_overlap_n": int(rows.training_scaffold_overlap.sum())})
        with np.load(run / "oof_predictions.npz", allow_pickle=False) as arrays:
            from sklearn.metrics import mean_squared_error, roc_auc_score

            for scheme, models in result["validation"].items():
                for name, values in models.items():
                    observed, predicted = arrays["y"], arrays[scheme + "__" + name]
                    key = "roc_auc" if spec["task"] == "classification" else "rmse"
                    computed = float(roc_auc_score(observed, predicted)) if key == "roc_auc" else float(
                        np.sqrt(mean_squared_error(observed, predicted)))
                    if not np.isclose(computed, values[key], rtol=0, atol=1e-10):
                        raise ValueError("Cached validation metric disagrees with actual OOF predictions")
                    validation.append({"endpoint": endpoint, "scheme": scheme, "model": name,
                                       "metric": key, "value": computed,
                                       "n": len(observed)})
        metrics.append(Metric(name=endpoint + "_valid_training_rows", value=result["spec"]["n_train_valid"],
                              population="public endpoint-specific training records"))
        sources.append(DataSource(name=endpoint, sha256=spec["sha256"], source_url=spec["source"],
                                  revision="public-source-method-reproduction-v1", derivation=spec["notes"],
                                  parent_sha256=spec.get("parent_sha256")))
        for name, media in [("predictions.csv", "text/csv"), ("effective_training.csv", "text/csv"),
                            ("rejected_training.csv", "text/csv"), ("rejected_query.csv", "text/csv"),
                            ("result.json", "application/json"), ("folds.json", "application/json")]:
            artifacts.append(save_artifact((run / name).read_bytes(), suffix=name.split(".")[-1],
                                           media_type=media, public_base_url=base,
                                           display_name=endpoint + "_" + name))
    candidate_frame, named_tables = toxicity_tables(candidate_predictions)
    artifacts.append(save_table(table(candidate_frame, "All mapped Figure4 candidate-endpoint predictions and domain/overlap flags"),
                                port=7341, display_name="figure4_candidate_endpoint_predictions.csv"))
    result = AnalysisResult(status="partial", answer=Answer(
        finding=f"Validated open-model profiles for {len(endpoints)} endpoints on the deposited Cannabis "
                "source cohort, with endpoint-specific paired validation, applicability and training overlap. "
                "Ten reviewed Figure4 connectivities are linked to their source-model predictions for every requested "
                "endpoint. These are newly reconstructed values, not original Syntelly outputs or a full final-paper cohort replay.",
        metrics=metrics, tables={"cohort_comparisons": table(pd.DataFrame(summaries), "Descriptive comparison; all query rows retained"),
                                "model_validation": table(pd.DataFrame(validation), "Pooled OOF metrics from independent fold fits"),
                                "paper2024_preparation_audit": table(pd.DataFrame(preparation_summaries),
                                    "Computed chemistry/conflict/aggregation audit; existing v1 models have NOT been retrained on these aggregates"),
                                **named_tables},
        limitations=[
            EvidenceNote(code="OPEN_METHOD_NOT_PROPRIETARY_WEIGHTS", message="Two fixed CatBoost-ECFP and "
                         "XGBoost-fragment models follow the existing ChemAgents open implementation on declared "
                         "public data. They are new fits, not proprietary Syntelly outputs or measured toxicity."),
            EvidenceNote(code="PAPER2024_METHOD_DIFFERENCES", message="The original 2024 paper, section 4.3 "
                         "and supplement S1, additionally describes source-aware duplicate aggregation, "
                         "element filtering, custom fragment selection and Optuna/GridSearch tuning. "
                         "The new preparation audit applies chemistry/conflict/aggregation rules separately; "
                         "it does not alter the training cohort of the returned frozen v1 predictions. "
                         "Train-only selector and documented search spaces are implemented, but these v1 models "
                         "have not been retuned with them. Collection-level source labels do not recover the "
                         "original constituent experimental sources. "
                         "Its independently validated outputs must not be labeled an exact replay of the "
                         "paper's tuned models or custom fragment dictionary."),
            EvidenceNote(code="SOURCE_COHORT_MISMATCH", message="Available source data contain 2749 unlabeled, "
                         "1680 pesticide-reference and 1491 AChE-inactive records. No silent replacement of the "
                         "paper's 5027/1709 whole-metabolome comparison or its final top ten is made."),
            EvidenceNote(code="ENDPOINT_DEPENDENCIES", message="hERG measures channel blockade rather than "
                         "general cardiotoxicity. DILI is drug-induced liver injury; the old Heracleum server "
                         "also used this same dataset as a broader hepatotoxicity proxy. These are not "
                         "independent biological endpoints. No separate hepatotoxicity result is invented."),
            EvidenceNote(code="APPLICABILITY_NOT_ACCURACY", message="kNN Tanimoto distances are normalized "
                         "with the training-reference mean + 2 SD and Gaussian mapping. The reported percentage "
                         "is a relative domain score, not a calibrated probability of correctness. No query "
                         "molecule is excluded by applicability. Identity and scaffold overlaps are explicit."),
            EvidenceNote(code="VALIDATION_SCOPE", message="identity_random keeps exact molecular identities "
                         "together; scaffold keeps empty acyclic Murcko cores together as well. These paired "
                         "validation runs do not replace the separately implemented original Supplement S3 replays."),
            EvidenceNote(code="AQUATIC_UNITS_UNVERIFIED", message="Aquatic predictions retain the source pLC50 "
                         "scale. No absolute concentration conversion or aquatic hazard class is inferred.")]),
        metadata=Provenance(study_id="cannabis_2026", paper_doi="10.3390/ijms27188000",
                            protocol_id="chemagents_open_toxicity_two_boosters_v1",
                            protocol_parameters={"endpoints": endpoints, "query_sha256": expected["query_sha256"],
                                                 "ensemble": "fixed arithmetic mean", "cv_folds": 5,
                                                 "paper2024_preparation_assumptions": ASSUMPTIONS,
                                                 "paper2024_preparation_applied_to_predictions": False},
                            sources=sources, software=result["spec"]["software"],
                            implementation_sha256=implementation_sha256(), reused_from_cache=reused), artifacts=artifacts)
    attach_mapping_provenance(result, base_url=base)
    result.artifacts.append(save_json(result.model_dump(mode="json"), public_base_url=base,
                                       display_name="cannabis_open_toxicity_profiles.json"))
    return result
