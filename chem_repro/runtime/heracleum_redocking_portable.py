"""Portable exact-Vina launcher over the immutable v1 chemical preparation code."""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path

from chem_repro.common import vina_binaries
from chem_repro.vendor import heracleum_redocking_worker as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--vina", type=Path, required=True)
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()
    software = {name: version(name) for name in core.PINS}
    if software != core.PINS:
        raise ValueError("Scientific preparation environment differs from the pinned protocol")
    binary, execution = vina_binaries.probe(args.vina)
    rows = list(csv.DictReader(args.table.open()))
    eligible = [r for r in rows if r["evaluated"] == "True"]
    if args.targets:
        if len(set(args.targets)) != len(args.targets) or set(args.targets) - {r["target"] for r in eligible}:
            raise ValueError("Require unique target identifiers from the evaluated panel")
        eligible = [r for r in eligible if r["target"] in args.targets]
    root = args.output.resolve()
    if root.exists() and not (root / "run_spec.json").exists() and any(root.iterdir()):
        raise ValueError("Nonempty output lacks an immutable run specification; refuse to relabel existing poses")
    root.mkdir(parents=True, exist_ok=True)
    spec = {"protocol": "public_crystal_redocking_reconstruction.v2", "fidelity": core.FIDELITY,
            "table_sha256": core.digest(args.table), "worker_sha256": core.digest(__file__),
            "preparation_worker_sha256": core.digest(core.__file__),
            "binary_policy_sha256": core.digest(vina_binaries.__file__),
            "binary_catalog_sha256": vina_binaries.CATALOG_SHA256,
            "binary_identity": binary, "execution_platform": execution,
            "software": software, "binary_sha256": binary["sha256"],
            "targets": [r["target"] for r in eligible],
            "author_exclusions": [r for r in rows if r["evaluated"] != "True"],
            "exhaustiveness": 64, "num_modes": 20, "energy_range": 4, "seed": 42, "cpu": 2}
    spec_path = root / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Existing run has a different immutable protocol/platform specification")
    core.write_json(spec_path, spec)
    receipts = []
    for row in eligible:
        print(json.dumps({"target": row["target"], "phase": "starting", "binary": binary["name"]}), flush=True)
        receipt = core.run_target(row, root, args.sources.resolve(), args.vina.resolve())
        receipts.append(receipt)
        summary = {"requested": len(eligible), "finished": len(receipts),
                   "computed": sum(r["status"] == "computed" for r in receipts),
                   "failed": sum(r["status"] == "failed" for r in receipts), "targets": receipts}
        core.write_json(root / "summary.json", summary)
        print(json.dumps({"target": row["target"], "status": receipt["status"],
                          "failure_phase": receipt.get("failure_phase"),
                          "best_top5_rmsd_A": receipt.get("pose_recovery", {}).get("best_top5_rmsd_A"),
                          "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
