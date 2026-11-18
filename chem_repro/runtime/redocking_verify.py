"""Recalculate fixed-frame native-pose RMSD in the pinned docking environment."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from chem_repro.vendor.heracleum_redocking_worker import FIDELITY, PINS, pose_recovery


def verify(directory):
    software = {name: version(name) for name in PINS}
    if software != PINS:
        raise ValueError("RMSD verifier software differs from the executed reconstruction")
    spec = json.loads((directory / "run_spec.json").read_text())
    if spec.get("fidelity") != FIDELITY:
        raise ValueError("Recorded preparation boundaries differ from the executed worker")
    summary = json.loads((directory / "summary.json").read_text())
    rows = []
    for target in summary["targets"]:
        name = target["target"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe target identifier")
        if target["status"] != "computed":
            continue
        result = pose_recovery(directory / name / "ligand_native_h.sdf", directory / name / "docked.pdbqt")
        if result != target["pose_recovery"]:
            raise ValueError(f"Recomputed per-pose RMSD or scores disagree for {name}")
        rows.append({"target": name, **result})
    return {"software": software, "verification": "recomputed_from_stored_native_and_docked_coordinates",
            "summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
            "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve()), allow_nan=False))
