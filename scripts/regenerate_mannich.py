"""Resumable source-protocol regeneration with one independently saved result per catalyst."""

import argparse
import json
from pathlib import Path

from chem_repro.common.datasets import implementation_sha256
from chem_repro.mannich.analysis import load
from chem_repro.mannich.conformers import regenerate_catalysts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", choices=["source_notebook", "geometry_verified"], default="source_notebook"
    )
    parser.add_argument("--filenames", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    filenames = args.filenames or sorted({Path(v).name for v in load().organocatalyst_3d_sdf})
    code_hash = implementation_sha256()
    totals = {
        "requested_catalysts": len(filenames),
        "completed_catalysts": 0,
        "generated_catalysts": 0,
        "point_stereo_contradictions": 0,
        "axial_geometry_matches": 0,
        "axial_annotations": 0,
    }
    records = []
    for name in filenames:
        path = args.output / (name + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            assert result["metadata"]["implementation_sha256"] == code_hash, (
                "Source changed; choose a new result directory"
            )
            assert result["metadata"]["protocol_id"] == "mannich.regeneration." + args.protocol + ".v1"
        else:
            print("START", name, flush=True)
            result = regenerate_catalysts(args.protocol, [name]).model_dump(mode="json")
            pending = path.with_suffix(".pending")
            pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            pending.replace(path)
        metrics = {m["name"]: m["value"] for m in result["answer"]["metrics"]}
        totals["completed_catalysts"] += 1
        for key in [
            "generated_catalysts",
            "point_stereo_contradictions",
            "axial_geometry_matches",
            "axial_annotations",
        ]:
            totals[key] += metrics[key]
        row = result["answer"]["tables"]["generation_records"]["rows"][0]
        records.append(row)
        print(
            json.dumps(
                {
                    "filename": name,
                    "status": row["generation_status"],
                    "rmsd": row["aligned_heavy_atom_rmsd_to_deposit_A"],
                    **totals,
                }
            ),
            flush=True,
        )
        (args.output / "progress.json").write_text(
            json.dumps(
                {
                    "kind": "actual_conformer_generation_not_coscientist_e2e",
                    "protocol": args.protocol,
                    "implementation_sha256": code_hash,
                    **totals,
                },
                indent=2,
            )
            + "\n"
        )
    (args.output / "all_generation_records.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "receipt.json").write_text((args.output / "progress.json").read_text())


if __name__ == "__main__":
    main()
