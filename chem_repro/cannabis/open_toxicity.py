"""Typed MCP handoff for the independently reproduced open toxicity supplement."""

import hashlib
import io
import json
import os
import subprocess
import zipfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

from ..common.artifacts import artifact_root, save_artifact
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest

REFERENCES = {
    ("ames", "random"): 0.9225,
    ("ames", "source_scaffold"): 0.877,
    ("daphnia", "random"): 1.026,
    ("daphnia", "source_scaffold"): 1.183,
    ("fathead", "random"): 0.788,
    ("fathead", "source_scaffold"): 0.947,
    ("reproductive", "random"): 0.586,
}


def expected_recipe_hashes(cohort="toxric"):
    root = Path(__file__).resolve().parents[1]
    worker = "open_tox_worker.py" if cohort == "toxric" else "open_tox_augmented_worker.py"
    paths = [root / "runtime" / worker, *sorted((root / "vendor/cannabis_open_tox").glob("*.py"))]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def validate_open_toxicity(endpoint="ames", scheme="random", cohort="toxric"):
    if (endpoint, scheme) not in REFERENCES:
        raise ValueError("This endpoint/scheme has no reported Supplement S3 reference")
    if cohort not in {"toxric", "toxric_ecotox"} or (
        cohort == "toxric_ecotox" and endpoint not in {"daphnia", "fathead"}
    ):
        raise ValueError("Augmented source cohorts exist only for the two aquatic endpoints")
    manifest = verified_manifest("cannabis_open_tox")
    filename = f"toxric_{endpoint}.csv" if cohort == "toxric" else f"{endpoint}_aug.csv"
    source = DATA_ROOT / "cannabis_open_tox" / filename
    expected_software = dict(
        line.split("==")
        for line in (DATA_ROOT / "cannabis_open_tox/worker_requirements.txt").read_text().splitlines()
        if line
    )
    root = Path(os.getenv("CHEM_REPRO_OPEN_TOX_RESULTS_DIR", str(artifact_root() / "open_tox_runs")))
    run = root / (
        f"{endpoint}_{scheme}" if cohort == "toxric" else f"{endpoint}_{scheme}_toxric_ecotox_exact"
    )
    output = run / "result.json"
    reused = output.exists()
    if not output.exists():
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(
                "Incomplete source replay artifacts exist. Verify the worker process before retrying; no duplicate training was started."
            )
        executable = os.getenv("CHEM_REPRO_TOX_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("Configure CHEM_REPRO_TOX_PYTHON with the pinned isolated worker environment")
        run.mkdir(parents=True, exist_ok=True)
        worker_name = "open_tox_worker.py" if cohort == "toxric" else "open_tox_augmented_worker.py"
        worker = Path(__file__).resolve().parents[1] / "runtime" / worker_name
        from ..common.worker_environment import worker_environment

        with (run / "worker.log").open("w") as log, worker_environment(worker.parents[1]) as environment:
            completed = subprocess.run(
                [
                    executable,
                    str(worker),
                    "--endpoint",
                    endpoint,
                    "--scheme",
                    scheme,
                    "--output",
                    str(run.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
            )
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError("Open toxicity replay did not complete; inspect the persisted worker log")
    saved = json.loads(output.read_text())
    spec = saved["run_spec"]
    if (
        spec["source_sha256"] != manifest["files"][source.name]["sha256"]
        or spec["endpoint"] != endpoint
        or spec["scheme"] != scheme
        or spec["software"] != expected_software
        or spec["source_code_sha256"] != expected_recipe_hashes(cohort)
    ):
        raise ValueError("Replay provenance differs from current pinned data, code or dependency contract")
    frame = pd.read_csv(source)
    metrics = saved["metrics"]
    classification = endpoint in {"ames", "reproductive"}
    score = (
        (lambda y, p: float(roc_auc_score(y, p)))
        if classification
        else (lambda y, p: float(np.sqrt(mean_squared_error(y, p))))
    )
    with np.load(run / "oof_predictions.npz", allow_pickle=False) as arrays:
        observed = arrays["observed"]
        if len(observed) != len(frame) or not np.allclose(observed, frame.y.to_numpy(), atol=1e-12, rtol=0):
            raise ValueError("Source cohort changed or rows were excluded; explicit row-level audit required")
        rows = []
        for model, reported in metrics["base_scores"].items():
            calculated = score(observed, arrays[model])
            if abs(calculated - reported) > 1e-12:
                raise ValueError("Saved base metric disagrees with saved OOF predictions")
            rows.append({"model": model, "metric": metrics["metric"], "value": calculated})
        averaged = np.column_stack([arrays[name] for name in metrics["base_scores"]]).mean(axis=1)
        for name, predictions, expected in [
            ("fixed_mean_blend", averaged, metrics["mean_blend"]),
            ("source_meta_stack", arrays["source_meta"], metrics["source_meta_stack"]),
        ]:
            value = score(observed, predictions)
            if abs(value - expected) > 1e-12:
                raise ValueError("Saved ensemble metric disagrees with saved predictions")
            rows.append({"model": name, "metric": metrics["metric"], "value": value})
    digits = 4 if (endpoint, scheme) == ("ames", "random") else 3
    quantum = Decimal(1).scaleb(-digits)
    reference = Decimal(str(REFERENCES[(endpoint, scheme)]))
    for row in rows:
        row["matches_published_rounding"] = (
            Decimal(str(row["value"])).quantize(quantum, rounding=ROUND_HALF_UP) == reference
        )
    matching = [row["model"] for row in rows if row["matches_published_rounding"]]
    folds = json.loads((run / "folds.json").read_text())
    leakage = [{k: v for k, v in row.items() if not k.endswith("_indices")} for row in folds["meta_folds"]]
    result = AnalysisResult(
        status="partial",
        answer=Answer(
            finding=f"Replayed the open {endpoint} {scheme} benchmark on {len(frame)} deposited {cohort} records. "
            f"The source meta-stack {metrics['metric']} is {metrics['source_meta_stack']:.4f}. "
            f"Published-precision matches: {', '.join(matching) or 'none'}. "
            "All base and ensemble results are retained; no model or weight was tuned to the reference.",
            metrics=[
                Metric(name="n_records", value=len(frame), population=cohort + " open benchmark"),
                Metric(
                    name="supplement_s3_reference",
                    value=REFERENCES[(endpoint, scheme)],
                    population=f"published {endpoint} {scheme} benchmark",
                    evidence_kind="paper_reported",
                ),
            ],
            tables={
                "computed_model_metrics": table(
                    pd.DataFrame(rows),
                    "Actual OOF scores, independently recomputed from persisted predictions",
                ),
                "meta_validation_audit": table(
                    pd.DataFrame(leakage),
                    "Test-label exposure through base-model feature construction; source meta splits ignore groups",
                ),
            },
            limitations=[
                EvidenceNote(
                    code="COHORT_AND_ESTIMATOR_COMPARABILITY",
                    message="Audited replays recover "
                    "the aquatic scaffold references on augmented cohorts (Daphnia614, fathead1100), "
                    "whereas random references match TOXRIC-only cohorts (345 and812). The matches also "
                    "use different aggregators for some endpoints. This supports a mixed-cohort explanation "
                    "but does not uniquely identify historical runs. A random/scaffold penalty requires "
                    "the same cohort and estimator. Do not silently change cohorts to match a reference.",
                ),
                EvidenceNote(
                    code="OPEN_MODELS_NOT_MAIN_TEXT",
                    message="These are the four independent open "
                    "reproduction endpoints from Supplement S3, not the Syntelly predictions used for the main "
                    "metabolome comparison or final candidate toxicology.",
                ),
                EvidenceNote(
                    code="SOURCE_STACK_NOT_NESTED",
                    message="The source fits meta CV to a globally "
                    "constructed base OOF matrix. Meta-training features can therefore depend on meta-test "
                    "labels through base-model training. This replay is not a strictly nested validation; "
                    "the source's leak-free wording is not adopted. No estimate of the resulting bias is claimed.",
                ),
                EvidenceNote(
                    code="SOURCE_SCAFFOLD_POLICY",
                    message="For source_scaffold, base models use "
                    "ring scaffolds with whole-molecule fallback for acyclic structures. The meta stage uses "
                    "ordinary unshuffled CV, not group-preserving folds. Source meta scores cannot be presented "
                    "as an independently scaffold-disjoint test.",
                ),
                EvidenceNote(
                    code="REFERENCE_MODEL_SELECTION",
                    message="The reported reference is shown "
                    "separately; a model is not selected to force agreement. In the reproductive replay the "
                    "reference aligns with CatBoost-ECFP rather than the source meta-stack.",
                ),
            ],
        ),
        metadata=provenance(
            "cannabis_open_tox",
            "cannabis.open_tox.source_replay.v1",
            endpoint=endpoint,
            scheme=scheme,
            cohort=cohort,
            estimator_threads=spec["threads"],
            worker_software=spec["software"],
            worker_code_sha256=spec["source_code_sha256"],
            feature_dimensions=metrics["feature_dimensions"],
            target_values="TOXRIC y column as deposited",
            regression_units="source pLC50 scale; no absolute concentration conversion inferred",
        ),
    )
    result.metadata.protocol_parameters["interface_software"] = dict(result.metadata.software)
    result.metadata.software.update(spec["software"])
    result.metadata.reused_from_cache = reused
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ["result.json", "run_spec.json", "folds.json", "oof_predictions.npz"]:
            archive.writestr(name, (run / name).read_bytes())
    result.artifacts.append(
        save_artifact(
            bundle.getvalue(),
            suffix="zip",
            media_type="application/zip",
            public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341"),
            display_name=f"cannabis_open_tox_{endpoint}_{scheme}_{cohort}.zip",
        )
    )
    return result
