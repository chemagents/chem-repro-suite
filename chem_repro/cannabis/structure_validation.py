"""Fixed-data Chemprop validation adapter, shared-output contracts and verifiable caches."""

import fcntl
import hashlib
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.artifacts import save_artifact
from ..common.classification import binary_metrics
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .analysis import metric, result

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "runtime/dmpnn_worker.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_structure(protocol="paired_cv", source_split_indices=None, max_epochs=60, include_models=False):
    if protocol not in {"paired_cv", "source_random"} or not 3 <= max_epochs <= 200:
        raise ValueError("Unknown protocol or invalid training budget")
    if protocol == "paired_cv" and source_split_indices is not None:
        raise ValueError("source_split_indices applies only to source_random")
    indices = list(range(30)) if source_split_indices is None else source_split_indices
    if protocol == "source_random" and (not indices or len(set(indices)) != len(indices) or
                                         any(not isinstance(i, int) or not 0 <= i < 30 for i in indices)):
        raise ValueError("Source split indices must be unique integers in 0..29")
    manifest = verified_manifest("cannabis")
    config = DATA_ROOT / "cannabis_dmpnn"
    verified_manifest("cannabis_dmpnn")
    python = os.getenv("CHEM_REPRO_CHEMPROP_PYTHON")
    if not python or not Path(python).is_file():
        raise RuntimeError("Set CHEM_REPRO_CHEMPROP_PYTHON to the pinned Chemprop CPU worker environment")
    expected_software = json.loads((config / "scientific_versions.json").read_text())
    probe = "import json; from importlib.metadata import version; print(json.dumps({n:version(n) for n in " + repr(list(expected_software)) + "}))"
    from ..common.worker_environment import worker_environment

    with worker_environment(ROOT) as environment:
        actual_software = json.loads(subprocess.check_output([python, "-c", probe], text=True, timeout=30, env=environment))
    if actual_software != expected_software:
        raise RuntimeError("Chemprop worker scientific packages differ from the pinned environment")
    code = {"runtime/dmpnn_worker.py": sha(WORKER), "common/dmpnn.py": sha(ROOT / "common/dmpnn.py"),
            "common/classification.py": sha(ROOT / "common/classification.py")}
    root = Path(os.getenv("CHEM_REPRO_STRUCTURE_RESULTS_DIR", "structure_validation_runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    summaries, folds, artifacts, runs = [], [], [], []
    reused = True
    for scheme in (["random", "scaffold"] if protocol == "paired_cv" else ["source_random"]):
        request = {"scheme": scheme, "source_sha256": manifest["files"]["data.csv"]["sha256"],
                   "code": code, "software": actual_software, "max_epochs": max_epochs,
                   "source_split_indices": indices if scheme == "source_random" else None}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        run = root / (scheme + "_" + key[:16])
        with (root / (key + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("The same structural validation is already running; no duplicate fit started") from exc
            if not (run / "result.json").exists():
                reused = False
                # The child inherits the lock, so an orphaned training process still prevents duplicate fits.
                with (root / (run.name + ".log")).open("a") as log, worker_environment(ROOT) as environment:
                    subprocess.run([python, str(WORKER), "--data-dir", str(DATA_ROOT / "cannabis"),
                                    "--output", str(run), "--scheme", scheme, "--folds", "5",
                                    "--split-indices", ",".join(map(str, indices)), "--max-epochs", str(max_epochs),
                                    "--patience", "15", "--batch-size", "50", "--threads", "2"],
                                   check=True, timeout=21600, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, pass_fds=(lock.fileno(),))
            saved = json.loads((run / "result.json").read_text())
            spec = saved["run_spec"]
            if any(spec[k] != request[k] for k in ["scheme", "source_sha256", "code", "software"]):
                raise ValueError("Cached structural model provenance differs")
            if spec["explicit_choices"]["max_epochs"] != max_epochs:
                raise ValueError("Cached training budget differs")
            for name, expected in saved["top_level_artifacts"].items():
                if sha(run / name) != expected:
                    raise ValueError("Structural result artifact digest mismatch")
            split_rows = []
            for path in sorted(run.glob("fold_*/result.json")):
                fit = json.loads(path.read_text())
                if fit["run_spec_sha256"] != sha(run / "run_spec.json"):
                    raise ValueError("Fold does not belong to this exact run")
                for name, expected in fit["artifacts"].items():
                    if sha(path.parent / name) != expected:
                        raise ValueError("Fold model or prediction digest mismatch")
                with np.load(path.parent / "predictions.npz", allow_pickle=False) as arrays:
                    recomputed = binary_metrics(arrays["y_test"], arrays["probabilities"], threshold=.5)
                expected_metrics = fit["metrics"]
                if set(recomputed) != set(expected_metrics) or any(
                    not np.isclose(recomputed[name], value, rtol=0, atol=1e-12)
                    if isinstance(value, float) else recomputed[name] != value
                    for name, value in expected_metrics.items()
                ):
                    raise ValueError("Fold metrics differ from actual saved held-out probabilities")
                split_rows.append(fit["split"])
                folds.append({"scheme": scheme, "fold": fit["split"]["fold"],
                              "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
                              "identity_train_test_overlap": fit["split"]["identity_train_test_overlap"],
                              "scaffold_train_test_overlap": fit["split"]["scaffold_train_test_overlap"], **recomputed})
            if len(split_rows) != saved["n_fits"]:
                raise ValueError("Completed fold count differs from the result")
            pooled = saved["pooled_oof_metrics"] or {}
            summary = {"scheme": scheme, "n_fits": saved["n_fits"], "n_labeled": saved["n_labeled"],
                       "n_query": saved["n_queried"], "pooled_oof_roc_auc": pooled.get("roc_auc"),
                       "pooled_oof_average_precision": pooled.get("average_precision"),
                       "pooled_oof_pr_auc_trapezoidal": pooled.get("pr_auc_trapezoidal")}
            for name, values in saved["fold_mean_std"].items():
                summary[name + "_fold_mean"], summary[name + "_fold_std"] = values["mean"], values["std"]
            summaries.append(summary)
            runs.append({"scheme": scheme, "cache_id": run.name, "run_spec": spec,
                         "source_diagnostics": saved["source_diagnostics"]})
            for name, media in [("result.json", "application/json"), ("run_spec.json", "application/json"),
                                ("heldout_predictions.csv", "text/csv"), ("query_predictions_by_fold.csv", "text/csv"),
                                ("fold_metrics.csv", "text/csv"), ("structure_audit.csv", "text/csv"),
                                ("descriptor_names.json", "application/json")]:
                artifacts.append(save_artifact((run / name).read_bytes(), suffix=name.rsplit(".", 1)[1],
                                               media_type=media, public_base_url=base, display_name=scheme + "_" + name))
            artifacts.append(save_artifact((json.dumps(split_rows, indent=2) + "\n").encode(), suffix="json",
                                           media_type="application/json", public_base_url=base,
                                           display_name=scheme + "_train_validation_test_partitions.json"))
            if include_models:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in run.glob("fold_*/*"):
                        if path.name in {"weights.pt", "scaler.json", "result.json", "training_curve.csv"}:
                            archive.write(path, str(path.relative_to(run)))
                    for name in ["run_spec.json", "descriptor_names.json", "structure_audit.csv"]:
                        archive.write(run / name, name)
                artifacts.append(save_artifact(buffer.getvalue(), suffix="zip", media_type="application/zip",
                                               public_base_url=base, display_name=scheme + "_models_and_scalers.zip"))
    reference = json.loads((config / "dmpnn_published_reference.json").read_text())["paired_kfold"]["rmt_rte_rdkit_only"]["by_split"]
    reference_rows = [{"scheme": scheme, "roc_auc": item["dmpnn"]["baseline_roc"],
                       "pr_auc_reported": item["dmpnn"]["baseline_pr"]} for scheme, item in reference.items()]
    output = result(
        "Chemprop directed-bond message passing was actually fitted and evaluated on the available source "
        "cohort with separate validation partitions and reloaded best checkpoints. Full held-out predictions "
        "and source IDs are exported. These new computations do not establish exact recovery of author folds or weights.",
        [metric("structural_model_fits", sum(r["n_fits"] for r in summaries)),
         metric("labeled_source_records", summaries[0]["n_labeled"]), metric("unlabeled_source_records", summaries[0]["n_query"])],
        {"validation_summary": table(pd.DataFrame(summaries), "Computed validation; pooled OOF only for disjoint K-fold schemes"),
         "fold_metrics": table(pd.DataFrame(folds), "Per-fold performance, early stopping and overlap diagnostics"),
         "paper_reference": table(pd.DataFrame(reference_rows), "Supplement S4 source-record values, not recomputed predictions",
                                   evidence_kind="paper_reported")},
        protocol="cannabis.chemprop_source_reconstruction.v1",
        notes=[EvidenceNote(code="DMPNN_FIDELITY_BOUNDARY", message="The paper configuration is depth3, hidden300, "
                            "dropout0.1, two total FFN linear layers, and mean/sum/norm aggregation. Chemprop2.2.1, "
                            "norm100, V2 atom features, train-only descriptor scaling/clipping, 60epochs/15patience, "
                            "validation ROC-AUC and specific split/seed choices are explicit reconstruction choices. "
                            "Final author checkpoints/partitions are unavailable. The paper reports3403standardized reference "
                            "records, but does not establish their exact membership in the DMPNN training run. This source uses3171labels."),
               EvidenceNote(code="SOURCE_GRAPH_VIEW", message="Stored source connectivity, charges and isotopes are "
                            "retained. Only stereochemical labels are explicitly removed. Charged structures are "
                            "reported, not automatically treated as errors. Scaffolds include the empty acyclic core."),
               EvidenceNote(code="PREDICTED_PESTICIDE_CLASS", message="Scores distinguish pesticide references from "
                            "AChE-inactive controls. They are not measured pesticidal activity or calibrated efficacy. "
                            "The0.5threshold is for evaluation; the paper0.70candidate screen is a separate operation.")],
        training_protocol=protocol, max_epochs=max_epochs, worker_runs=runs)
    output.status = "partial"
    output.metadata.sources.extend(provenance("cannabis_dmpnn", "chemprop.source_configuration").sources)
    output.metadata.reused_from_cache = reused
    output.artifacts.extend(artifacts)
    return output
