"""Executed public-crystal redocking, separate from the authors' Table S1a observations."""

import csv
import fcntl
import hashlib
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from ..common import vina_binaries
from ..common.artifacts import save_artifact, save_json
from ..common.contracts import AnalysisResult, Answer, DataSource, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from ..common.docking import pose_recovery_summary
from ..common.processes import run_worker
from ..common.worker_environment import worker_environment

PACKAGE = Path(__file__).resolve().parents[1]
WORKER = PACKAGE / "vendor/heracleum_redocking_worker.py"
PORTABLE_WORKER = PACKAGE / "runtime/heracleum_redocking_portable.py"
VERIFIER = PACKAGE / "runtime/redocking_verify.py"
VINA_SHA256 = "de9402701fba0f69bce53aec94254a08a7044c7d1d459958896022a7f98b073d"
SOFTWARE = {"meeko": "0.8.0", "rdkit": "2026.3.3", "numpy": "2.2.6", "scipy": "1.15.3", "gemmi": "0.7.5"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(directory, name, expected):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Unsafe cached artifact name")
    path = directory / name
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f"Cached artifact missing or changed: {name}")
    return path


def structural_source(original_path, expected, pdb):
    """Resolve relocated inputs by explicit root and checksum, not by a filename alone."""
    original = Path(original_path)
    configured = os.getenv("CHEM_REPRO_REDOCKING_SOURCES_DIR")
    if configured:
        if len(pdb) != 4 or not pdb.isalnum():
            raise ValueError("Unexpected fixed-panel PDB identifier")
        directory = Path(configured).resolve() / pdb.lower()
    else:
        directory = original.parent
    path = checked_file(directory, original.name, expected)
    if configured and not path.resolve().is_relative_to(Path(configured).resolve()):
        raise ValueError("Resolved structural source escapes the configured source root")
    receipt_path = path.with_suffix(path.suffix + ".source.json")
    if receipt_path.is_symlink():
        raise ValueError("Source receipt must be a regular local file")
    receipt = json.loads(receipt_path.read_text())
    if receipt["sha256"] != expected:
        raise ValueError("Public source receipt checksum differs from the executed input")
    return path, receipt_path, receipt


def inspect_run(directory):
    """Validate all target outcomes, including failed preparation, before reading their metrics."""
    verified_manifest("heracleum")
    reference = DATA_ROOT / "heracleum/redocking_table_s1a.csv"
    expected_rows = list(csv.DictReader(reference.open()))
    eligible = [row for row in expected_rows if row["evaluated"] == "True"]
    spec = json.loads((directory / "run_spec.json").read_text())
    expected = {"table_sha256": sha(reference), "software": SOFTWARE,
                "targets": [r["target"] for r in eligible],
                "author_exclusions": [r for r in expected_rows if r["evaluated"] != "True"],
                "exhaustiveness": 64, "num_modes": 20, "energy_range": 4, "seed": 42, "cpu": 2}
    if spec.get("protocol") == "public_crystal_redocking_reconstruction.v1":
        expected.update(worker_sha256=sha(WORKER), binary_sha256=VINA_SHA256)
    elif spec.get("protocol") == "public_crystal_redocking_reconstruction.v2":
        vina_binaries.validate_recorded(spec["binary_identity"], spec["execution_platform"])
        expected.update(worker_sha256=sha(PORTABLE_WORKER), preparation_worker_sha256=sha(WORKER),
                        binary_policy_sha256=sha(Path(vina_binaries.__file__)),
                        binary_catalog_sha256=vina_binaries.CATALOG_SHA256,
                        binary_sha256=spec["binary_identity"]["sha256"])
    else:
        raise ValueError("Unknown recorded public-redocking protocol")
    if any(spec.get(k) != value for k, value in expected.items()):
        raise ValueError("Cached redocking protocol, population, software or executed code differs")
    summary = json.loads((directory / "summary.json").read_text())
    if (summary.get("requested") != len(eligible) or summary.get("finished") != len(eligible)
            or len(summary.get("targets", [])) != len(eligible)):
        raise ValueError("Redocking panel is incomplete; do not start a duplicate worker or omit pending targets")
    if [r.get("target") for r in summary["targets"]] != spec["targets"]:
        raise ValueError("Target identity/order differs from the declared panel")
    for row, outcome in zip(eligible, summary["targets"]):
        target_dir = directory / row["target"]
        receipt = json.loads((target_dir / "receipt.json").read_text())
        if receipt != outcome or receipt.get("published_reference") != row:
            raise ValueError("Summary differs from an original target receipt or its published reference")
        if outcome.get("status") not in {"computed", "failed"}:
            raise ValueError("A target is not terminal")
        for name, expected_hash in outcome["output_sha256"].items():
            checked_file(target_dir, name, expected_hash)
        if outcome["status"] == "computed":
            required = {"ligand_native_h.sdf", "ligand.pdbqt", "receptor.pdbqt", "docked.pdbqt", "vina.operation.json"}
            if not required <= set(outcome["output_sha256"]):
                raise ValueError("Computed target lacks native coordinates, prepared structures or execution receipt")
            operation = json.loads((target_dir / "vina.operation.json").read_text())
            if operation.get("exit_code") != 0 or not outcome["pose_recovery"].get("no_superposition"):
                raise ValueError("Computed target has no successful fixed-frame docking execution")
        elif not outcome.get("failure_phase") or not outcome.get("error"):
            raise ValueError("Failed target lacks its preparation/execution failure record")
    if summary["computed"] != sum(r["status"] == "computed" for r in summary["targets"]):
        raise ValueError("Computed target count differs from per-target records")
    if summary["failed"] != sum(r["status"] == "failed" for r in summary["targets"]):
        raise ValueError("Failed target count differs from per-target records")
    return spec, summary


def configuration_missing(names):
    return AnalysisResult(status="unsupported", answer=Answer(
        finding="Public-crystal redocking requires the declared docking worker environment and output directory.",
        limitations=[EvidenceNote(code="REDOCKING_CONFIGURATION_REQUIRED", message="Configure " + ", ".join(names)
                                  + ". Published Table S1a aggregation remains separately available; no new docking was run.")]),
        metadata=provenance("heracleum", "heracleum.public_redocking_configuration.v1"))


def reconstruct(include_structures=True):
    """Run the fixed panel once, or verify existing poses; do not force failed structures to prepare."""
    required = ["CHEM_REPRO_DOCKING_PYTHON", "CHEM_REPRO_REDOCKING_RESULTS_DIR"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return configuration_missing(missing)
    python = Path(os.environ[required[0]])
    if not python.is_file():
        raise ValueError("Configured docking Python is unavailable")
    directory = Path(os.environ[required[1]]).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    reused = directory.exists()
    lock_path = directory.with_name(directory.name + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This redocking panel is already running; no duplicate started") from exc
        if not directory.exists():
            missing = [name for name in ["CHEM_REPRO_VINA_BINARY", "CHEM_REPRO_REDOCKING_SOURCES_DIR"] if not os.getenv(name)]
            if missing:
                return configuration_missing(missing)
            binary = Path(os.environ["CHEM_REPRO_VINA_BINARY"])
            if not binary.is_file():
                raise ValueError("Configured Vina binary is missing")
            vina_binaries.identity(sha(binary))
            command = [str(python), str(PORTABLE_WORKER), "--table", str(DATA_ROOT / "heracleum/redocking_table_s1a.csv"),
                       "--output", str(directory), "--sources", os.environ["CHEM_REPRO_REDOCKING_SOURCES_DIR"],
                       "--vina", str(binary)]
            with worker_environment(PACKAGE) as environment:
                run_worker(command, log_path=directory.with_name(directory.name + ".log"), timeout=24 * 3600,
                           pass_fds=(lock.fileno(),), env=environment)
        spec, summary = inspect_run(directory)
        with worker_environment(PACKAGE) as environment:
            verification = subprocess.run([str(python), str(VERIFIER), "--run-dir", str(directory)],
                                          capture_output=True, text=True, timeout=600, check=True,
                                          env=environment)
        verified = json.loads(verification.stdout)
        if verified["summary_sha256"] != sha(directory / "summary.json"):
            raise ValueError("Target results changed during coordinate verification")
        computed = {row["target"]: row for row in verified["results"]}
        if set(computed) != {r["target"] for r in summary["targets"] if r["status"] == "computed"}:
            raise ValueError("Coordinate verification does not cover all computed targets")
        rows, poses = [], []
        for saved in summary["targets"]:
            target = saved["target"]
            actual = computed.get(target)
            reported = saved["published_reference"]
            rows.append({"target": target, "pdb": saved["pdb"], "ligand": saved["ligand_code"],
                         "status": saved["status"], "failure_phase": saved.get("failure_phase"),
                         "failure_message": saved.get("error"),
                         "computed_top1_rmsd_A": actual["top1_rmsd_A"] if actual else None,
                         "computed_best_top5_rmsd_A": actual["best_top5_rmsd_A"] if actual else None,
                         "computed_first_recovery_rank_any_pose": actual["first_recovery_rank"] if actual else None,
                         "computed_first_recovery_rank_top5": actual["first_recovery_rank"] if actual and actual["first_recovery_rank"] is not None and actual["first_recovery_rank"] <= 5 else None,
                         "computed_top1_score_kcal_mol": actual["computed_poses"][0]["vina_score_kcal_mol"] if actual else None,
                         "paper_best_top5_rmsd_A": float(reported["best_top5_rmsd_A"]),
                         "paper_top1_score_kcal_mol": float(reported["vina_score_kcal_mol"]),
                         "paper_first_recovery_rank": int(reported["first_recovery_rank"]) if reported["first_recovery_rank"] else None})
            if actual:
                poses.extend({"target": target, **row} for row in actual["computed_poses"])
        n = len(computed)
        if n:
            records = pd.DataFrame([{
                "target": row["target"], "evaluated": True,
                "best_top5_rmsd_A": row["computed_best_top5_rmsd_A"],
                "first_recovery_rank": row["computed_first_recovery_rank_top5"], "exclusion_reason": "",
            } for row in rows if row["status"] == "computed"])
            recovery_frame, _, _, median_rmsd = pose_recovery_summary(records, rmsd_threshold=2.0)
            recovery_frame = recovery_frame.rename(columns={"n_evaluated": "n_with_computed_poses",
                                                            "fraction_recovered": "fraction_among_computed_targets"})
        else:
            median_rmsd = None
            recovery_frame = pd.DataFrame([{"top_k": k, "n_recovered": 0, "n_with_computed_poses": 0,
                                           "fraction_among_computed_targets": None} for k in range(1, 6)])
        recovery_frame["n_requested"] = len(rows)
        recovery_frame["n_without_computed_poses"] = len(rows) - n
        recovery = recovery_frame.to_dict("records")
        base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7342")
        artifacts = [save_json(verified, public_base_url=base, display_name="redocking_coordinate_verification.json"),
                     save_json(spec, public_base_url=base, display_name="redocking_run_spec.json")]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            if spec["protocol"].endswith(".v1"):
                archive.write(WORKER, "executed_worker.py")
            else:
                archive.write(PORTABLE_WORKER, "executed_worker.py")
                archive.write(WORKER, "preparation_worker_v1.py")
                archive.write(Path(vina_binaries.__file__), "vina_binary_policy.py")
                archive.write(vina_binaries.CATALOG, "vina_binary_catalog.json")
                for subdirectory in ["", "common", "vendor", "runtime"]:
                    prefix = "source/chem_repro/" + (subdirectory + "/" if subdirectory else "")
                    archive.writestr(prefix + "__init__.py", "")
                archive.write(PORTABLE_WORKER, "source/chem_repro/runtime/heracleum_redocking_portable.py")
                archive.write(WORKER, "source/chem_repro/vendor/heracleum_redocking_worker.py")
                archive.write(Path(vina_binaries.__file__), "source/chem_repro/common/vina_binaries.py")
                archive.write(vina_binaries.CATALOG, "source/chem_repro/data/heracleum/vina_binary_catalog.json")
                archive.write(DATA_ROOT / "heracleum/redocking_table_s1a.csv", "source/redocking_table_s1a.csv")
                archive.writestr("source/requirements.txt", "".join(f"{name}=={value}\n" for name, value in sorted(SOFTWARE.items())))
                archive.writestr("source/REPLAY.txt", "Use Python3.12 and requirements.txt. Obtain an accepted Vina asset and verify its SHA256.\n"
                    "From source/: python -m chem_repro.runtime.heracleum_redocking_portable --table redocking_table_s1a.csv "
                    "--sources ../source_inputs --output ../new_run --vina /path/to/verified/vina\n"
                    "source_inputs is included only when include_structures=True. New execution records its own binary/platform.\n"
                    "This is public-crystal reconstruction, not recovery of author-prepared inputs.\n")
            archive.write(VERIFIER, "coordinate_verifier.py")
            archive.write(directory / "summary.json", "summary.json")
            archive.write(directory / "run_spec.json", "run_spec.json")
            raw_members = {}
            for saved in summary["targets"]:
                target = saved["target"]
                archive.write(directory / target / "receipt.json", target + "/receipt.json")
                for name, expected in saved["output_sha256"].items():
                    path = checked_file(directory / target, name, expected)
                    if include_structures or path.suffix in {".json", ".log"}:
                        archive.write(path, target + "/" + name)
                if include_structures:
                    for original, expected in saved.get("source_sha256", {}).items():
                        path, receipt_path, _ = structural_source(original, expected, saved["pdb"])
                        member = "source_inputs/" + saved["pdb"].lower() + "/" + path.name
                        if member in raw_members and raw_members[member] != expected:
                            raise ValueError("Conflicting source bytes at one archive path")
                        if member not in raw_members:
                            archive.write(path, member)
                            archive.write(receipt_path, member + ".source.json")
                            raw_members[member] = expected
        artifacts.append(save_artifact(buffer.getvalue(), suffix="zip", media_type="application/zip", public_base_url=base,
                                       display_name="redocking_structures_and_audits.zip" if include_structures else "redocking_audits.zip"))
        revision = "v1" if spec["protocol"].endswith(".v1") else "v2"
        metadata = provenance("heracleum", "heracleum.public_crystal_redocking." + revision, execution_kind="public_crystal_reconstruction",
                              fixed_receptor_frame=True, symmetry_aware=True, ligand_superposition=False,
                              rmsd_threshold_A=2.0, source_protocol=spec, include_structures=include_structures)
        metadata.reused_from_cache = reused
        metadata.protocol_parameters["coordinate_verifier_sha256"] = sha(VERIFIER)
        metadata.protocol_parameters["worker_software"] = verified["software"]
        # Public source receipts preserve their actual original URLs and hashes without treating
        # prepared coordinates as the unavailable author inputs.
        sources = {}
        for saved in summary["targets"]:
            for path_text, expected in saved.get("source_sha256", {}).items():
                path, _, receipt = structural_source(path_text, expected, saved["pdb"])
                sources[expected] = DataSource(name=path.name, sha256=expected, source_url=receipt["url"],
                                               revision="RCSB retrieved " + receipt["retrieved_at"])
        metadata.sources.extend(sources.values())
        recovery_finding = (
            f"Recovery among computed targets is {recovery[0]['n_recovered']}/{n} top-1 and "
            f"{recovery[-1]['n_recovered']}/{n} top-5 at 2 Å. "
            if n else "No pose-recovery fraction can be computed because there are no verified poses. "
        )
        return AnalysisResult(status="partial", answer=Answer(
            finding=f"Public-crystal preparation and Vina redocking were attempted for all {len(rows)} eligible targets. "
                    f"{n} have coordinate-verified poses; {len(rows)-n} have explicit preparation/execution failures. "
                    + recovery_finding + "Missing poses are not counted as docking failures. "
                    "This reconstruction uses different, fully declared preparation and is not an exact replay of author-prepared files.",
            metrics=[Metric(name="requested_targets", value=len(rows), population="42 Table S1a evaluated targets"),
                     Metric(name="computed_pose_targets", value=n, population="42 attempted targets"),
                     Metric(name="targets_without_computed_poses", value=len(rows)-n, population="42 attempted targets"),
                     Metric(name="median_computed_best_top5_rmsd", value=median_rmsd,
                            unit="angstrom", population=f"only {n} targets with computed poses")],
            tables={"target_comparison": table(pd.DataFrame(rows), "Computed columns are new results; paper_* columns are source-reported references. Null means unavailable, never zero RMSD."),
                    "pose_recovery": table(pd.DataFrame(recovery), "Subset denominators and all42attempts are both retained; unavailable preparations are not pose-recovery failures."),
                    "per_pose_results": table(pd.DataFrame(poses, columns=["target", "rank", "vina_score_kcal_mol", "symmetry_aware_heavy_atom_rmsd_A"]), "All computed ranked poses, fixed receptor frame without ligand alignment"),
                    "author_exclusions": table(pd.DataFrame(spec["author_exclusions"]), "Two author exclusions retained separately from our preparation failures", "source_record")},
            limitations=[EvidenceNote(code="RECONSTRUCTION_NOT_AUTHOR_INPUTS", message=spec["fidelity"]),
                         EvidenceNote(code="INCOMPLETE_POSE_COVERAGE", message="Preparation failures remain in the42targetcoverage denominator; subset recovery cannot be extrapolated to the full panel."),
                         EvidenceNote(code="NO_BIOLOGICAL_VALIDATION", message="Pose recovery does not validate affinity, inhibition or toxicity. Raw active/inactive control records remain unavailable.")]),
            metadata=metadata, artifacts=artifacts)
