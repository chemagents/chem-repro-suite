"""Check serialized regenerated SDFs and compare the source versus nominated axes."""

import argparse
import io
import json
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from rdkit import Chem

from chem_repro.common.artifacts import artifact_root
from chem_repro.common.axial import inspect_axial_geometry
from chem_repro.mannich.stereochemistry import compare_stereochemistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for file in sorted(args.results.glob("*.sdf.json")):
        result = json.loads(file.read_text())
        row = result["answer"]["tables"]["generation_records"]["rows"][0]
        artifact = result["artifacts"][0]
        data = (artifact_root() / Path(urlparse(artifact["download_url"]).path).name).read_bytes()
        assert sha256(data).hexdigest() == artifact["sha256"] and len(data) == artifact["size_bytes"]
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            sdf_names = [name for name in archive.namelist() if name.endswith(".sdf")]
            if not row["generated"]:
                assert not sdf_names
                continue
            assert sdf_names == [row["filename"]]
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(archive.read(sdf_names[0])), removeHs=False)
            molecule = next(supplier)
            assert molecule is not None and next(supplier, None) is None
            assert (
                molecule.GetConformer().Is3D() and np.isfinite(molecule.GetConformer().GetPositions()).all()
            )
            checks = compare_stereochemistry(Chem.MolFromSmiles(row["source_smiles"]), molecule)
            assert checks["connectivity_match"] == row["graph_matches"]
            assert checks["specified_source_stereo_match"] == row["specified_point_stereo_matches"]
            record = {
                "filename": row["filename"],
                "graph_matches": checks["connectivity_match"],
                "point_stereo_matches": checks["specified_source_stereo_match"],
                "source_axial": row["source_axial_configuration"],
                "axial_status": "not_applicable",
                "source_axis_matches_nominated": None,
            }
            if row["source_axial_configuration"] in {"R", "S"}:
                inspected = inspect_axial_geometry(molecule)
                selected = inspected["selected_axis"]
                if selected:
                    prior = row["axial_geometry"]["selected_axis"]
                    assert prior is not None and selected["axial_rs"] == prior["axial_rs"], (
                        "Serialization changed axial assignment"
                    )
                    matches = selected["axial_rs"] == row["source_axial_configuration"]
                    conditional = selected["status"] != "assigned_on_nominated_axis"
                    record["axial_status"] = ("conditional_" if conditional else "") + (
                        "matches" if matches else "disagrees"
                    )
                    source_axis = row["diagnostics"]["source_diagnostics"].get("axis")
                    record["source_axis_matches_nominated"] = (
                        set(source_axis) == set(selected["axis_atoms"]) if source_axis else None
                    )
                else:
                    record["axial_status"] = inspected["selection_policy"]
            records.append(record)
    receipt = {
        "kind": "serialized_SDF_roundtrip_verification",
        "structures_verified": len(records),
        "graph_contradictions": sum(not r["graph_matches"] for r in records),
        "point_stereo_contradictions": sum(not r["point_stereo_matches"] for r in records),
        "axial_status_counts": dict(Counter(r["axial_status"] for r in records)),
        "source_axis_mismatches": sum(r["source_axis_matches_nominated"] is False for r in records),
        "records": records,
    }
    (args.results / "serialized_sdf_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print({k: v for k, v in receipt.items() if k != "records"}, flush=True)


if __name__ == "__main__":
    main()
