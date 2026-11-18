import csv
import hashlib
import json
import os
import subprocess
import sys

import pytest

from chem_repro.common.datasets import DATA_ROOT
from chem_repro.common.processes import run_worker
from chem_repro.heracleum import redocking


def failed_panel(tmp_path):
    reference = DATA_ROOT / "heracleum/redocking_table_s1a.csv"
    records = list(csv.DictReader(reference.open()))
    eligible = [r for r in records if r["evaluated"] == "True"]
    spec = {"protocol": "public_crystal_redocking_reconstruction.v1", "fidelity": "test fixture",
            "table_sha256": redocking.sha(reference), "worker_sha256": redocking.sha(redocking.WORKER),
            "software": redocking.SOFTWARE, "binary_sha256": redocking.VINA_SHA256,
            "targets": [r["target"] for r in eligible], "author_exclusions": [r for r in records if r["evaluated"] != "True"],
            "exhaustiveness": 64, "num_modes": 20, "energy_range": 4, "seed": 42, "cpu": 2}
    outcomes = []
    for row in eligible:
        directory = tmp_path / row["target"]
        directory.mkdir()
        log = b"Strict preparation could not match a residue template.\n"
        (directory / "receptor_preparation.log").write_bytes(log)
        receipt = {"target": row["target"], "published_reference": row, "status": "failed",
                   "failure_phase": "receptor_preparation", "error": "test fixture",
                   "output_sha256": {"receptor_preparation.log": hashlib.sha256(log).hexdigest()}}
        (directory / "receipt.json").write_text(json.dumps(receipt))
        outcomes.append(receipt)
    summary = {"requested": 42, "finished": 42, "computed": 0, "failed": 42, "targets": outcomes}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "run_spec.json").write_text(json.dumps(spec))
    return spec, summary


def test_failures_are_terminal_coverage_not_computed_pose_results(tmp_path):
    failed_panel(tmp_path)
    spec, result = redocking.inspect_run(tmp_path)
    assert len(spec["targets"]) == 42 and len(spec["author_exclusions"]) == 2
    assert result["computed"] == 0 and result["failed"] == result["finished"] == 42


@pytest.mark.parametrize("change", ["partial", "duplicate_target", "worker", "artifact", "receipt", "counts"])
def test_incomplete_stale_or_inconsistent_panel_is_refused(tmp_path, change):
    spec, summary = failed_panel(tmp_path)
    if change == "partial":
        summary["finished"] = 41
    elif change == "duplicate_target":
        summary["targets"][-1] = summary["targets"][0]
    elif change == "worker":
        spec["worker_sha256"] = "0" * 64
    elif change == "artifact":
        (tmp_path / spec["targets"][0] / "receptor_preparation.log").write_text("changed")
    elif change == "receipt":
        summary["targets"][0]["failure_phase"] = "docking"
    else:
        summary["computed"] = 1
    (tmp_path / "run_spec.json").write_text(json.dumps(spec))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        redocking.inspect_run(tmp_path)


def test_cached_paths_cannot_escape_or_be_symlinks(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"real")
    checksum = redocking.sha(source)
    link = tmp_path / "link"
    link.symlink_to(source)
    for name in ["../source", str(source), "link"]:
        with pytest.raises(ValueError):
            redocking.checked_file(tmp_path, name, checksum)


def test_missing_configuration_never_claims_docking_execution(monkeypatch):
    monkeypatch.delenv("CHEM_REPRO_DOCKING_PYTHON", raising=False)
    monkeypatch.delenv("CHEM_REPRO_REDOCKING_RESULTS_DIR", raising=False)
    result = redocking.reconstruct()
    assert result.status == "unsupported" and not result.artifacts
    assert "no new docking" in result.answer.limitations[0].message


@pytest.mark.skipif(os.name != "posix", reason="Scientific workers currently use POSIX process groups")
def test_worker_timeout_also_terminates_descendants(tmp_path):
    script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
    log = tmp_path / "worker.log"
    with pytest.raises(subprocess.TimeoutExpired):
        run_worker([sys.executable, "-c", script], log_path=log, timeout=2)
    child_pid = int(log.read_text().strip())
    result = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True, check=False)
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z")
