"""Controlled public-crystal reconstruction of Table S1a; never an author-input replay.

Run with the isolated pinned docking Python. Every target ends with a receipt,
including source, preparation and geometry failures. Original files are immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import gemmi
import numpy as np
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


PINS = {"meeko": "0.8.0", "rdkit": "2026.3.3", "numpy": "2.2.6",
        "scipy": "1.15.3", "gemmi": "0.7.5"}
VINA_SHA256 = "de9402701fba0f69bce53aec94254a08a7044c7d1d459958896022a7f98b073d"
FIDELITY = (
    "Public-crystal reconstruction with the published box/search settings; original prepared "
    "author receptor/ligand files are unavailable. First deposited model, all polymer chains, "
    "blank/A alternate positions; waters, ligands and external cofactors removed. Polymer "
    "modified residues are retained for strict Meeko preparation. No deletion of bad residues, "
    "no repaired loops, no ligand minimization or alignment, no pH/tautomer equivalence claimed."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download(url, path):
    if path.exists():
        receipt = json.loads(path.with_suffix(path.suffix + ".source.json").read_text())
        if receipt["url"] != url or receipt["sha256"] != digest(path):
            raise ValueError("Cached public source differs from its receipt")
        return receipt
    request = urllib.request.Request(url, headers={"User-Agent": "ChemRepro/0.1 (scientific reproduction)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                if not payload or len(payload) > 50_000_000:
                    raise ValueError("Empty or unexpectedly large structural source")
                receipt = {"url": url, "resolved_url": response.url,
                           "retrieved_at": datetime.now(timezone.utc).isoformat(),
                           "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    path.write_bytes(payload)
    write_json(path.with_suffix(path.suffix + ".source.json"), receipt)
    return receipt


def retained_atoms(residue):
    return [a for a in residue if a.altloc in {"\x00", " ", "A"}]


def select_ligand(structure, code, center, size):
    candidates = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name != code:
                continue
            atoms = [a for a in retained_atoms(residue) if a.element.atomic_number > 1]
            if not atoms:
                continue
            coordinates = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
            inside = bool(np.all(np.abs(coordinates - center) <= size / 2 + .001))
            candidates.append({"chain": chain.name, "resnum": residue.seqid.num,
                               "insertion_code": residue.seqid.icode.strip(), "code": code,
                               "n_heavy_atoms": len(atoms), "inside_published_box": inside,
                               "center_distance_A": float(np.linalg.norm(coordinates.mean(0) - center)),
                               "coordinates": coordinates.tolist(),
                               "atomic_numbers": [a.element.atomic_number for a in atoms]})
    inside = [c for c in candidates if c["inside_published_box"]]
    if len(inside) != 1:
        raise ValueError(f"Require one unambiguous native ligand fully in the published box: {candidates}")
    return inside[0], candidates


def receptor_pdb(structure, destination):
    receptor = gemmi.Structure()
    model = gemmi.Model("1")
    excluded = Counter()
    modified = []
    for chain in structure[0]:
        new_chain = gemmi.Chain(chain.name)
        for residue in chain:
            if residue.entity_type != gemmi.EntityType.Polymer:
                excluded[residue.name] += len(residue)
                continue
            if len(chain.name) != 1 or len(residue.name) > 3 or not -999 <= residue.seqid.num <= 9999:
                raise ValueError("Polymer identifiers cannot be represented losslessly in PDB")
            kept = residue.clone()
            for i in reversed(range(len(kept))):
                if kept[i].altloc not in {"\x00", " ", "A"}:
                    del kept[i]
            names = [atom.name for atom in kept]
            if len(set(names)) != len(names):
                raise ValueError("Duplicate blank/A atom names; alternate-position selection ambiguous")
            if residue.het_flag == "H":
                modified.append({"chain": chain.name, "residue": str(residue.seqid), "name": residue.name})
            if not len(kept):
                raise ValueError("Polymer residue has no blank/A atoms; do not silently drop it")
            new_chain.add_residue(kept)
        if len(new_chain):
            model.add_chain(new_chain)
    if not len(model):
        raise ValueError("No retained polymer chains")
    receptor.add_model(model)
    receptor.write_pdb(str(destination))
    # The PDB conversion may change formatting but must preserve heavy atom positions and identities.
    reread = gemmi.read_structure(str(destination))
    def records(s):
        return [(c.name, str(r.seqid), r.name, a.name, a.element.name,
                 round(a.pos.x, 3), round(a.pos.y, 3), round(a.pos.z, 3))
                for c in s[0] for r in c for a in r if a.element.atomic_number > 1]
    if records(receptor) != records(reread):
        raise ValueError("PDB serialization changed receptor atom identities or coordinates")
    return {"retained_polymer_chains": [c.name for c in receptor[0]],
            "removed_nonpolymer_atom_counts": dict(excluded), "retained_modified_polymer_residues": modified,
            "receptor_heavy_atoms": len(records(receptor)), "coordinate_roundtrip_verified": True}


def verify_native_ligand(sdf, selected, out):
    molecules = list(Chem.SDMolSupplier(str(sdf), removeHs=False))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError("Ligand instance SDF must contain one valid molecular graph")
    mol = molecules[0]
    indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    coordinates = np.asarray(mol.GetConformer().GetPositions())[indices]
    atomic_numbers = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in indices])
    reference = np.asarray(selected["coordinates"])
    if coordinates.shape != reference.shape:
        raise ValueError("Ligand instance SDF and crystal have different heavy atom counts")
    cost = np.linalg.norm(coordinates[:, None, :] - reference[None, :, :], axis=-1)
    cost[atomic_numbers[:, None] != np.array(selected["atomic_numbers"])[None, :]] = 1e6
    ii, jj = linear_sum_assignment(cost)
    maximum = float(cost[ii, jj].max())
    if maximum > .002:
        raise ValueError(f"Native SDF is not in the crystal coordinate frame: {maximum}")
    hydrogenated = Chem.AddHs(mol, addCoords=True)
    if not np.array_equal(np.asarray(hydrogenated.GetConformer().GetPositions())[indices], coordinates):
        raise ValueError("Hydrogen addition moved native heavy atom coordinates")
    with Chem.SDWriter(str(out)) as writer:
        writer.write(hydrogenated)
    return {"sdf_crystal_max_heavy_distance_A": maximum,
            "heavy_coordinates_preserved": True, "heavy_atoms": len(indices),
            "native_isomeric_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)}


def pose_recovery(native_sdf, poses):
    reference = Chem.RemoveHs(next(iter(Chem.SDMolSupplier(str(native_sdf), removeHs=False))))
    molecules = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(poses.read_text()))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError("Require one ligand graph reconstructed from PDBQT SMILES IDX")
    probe = Chem.RemoveHs(molecules[0])
    if Chem.MolToSmiles(probe, isomericSmiles=True) != Chem.MolToSmiles(reference, isomericSmiles=True):
        raise ValueError("Prepared/docked graph or stereochemical labels differ from native reference")
    maps = reference.GetSubstructMatches(probe, uniquify=False, useChirality=True, maxMatches=100000)
    if not maps or len(maps) >= 100000 or any(len(m) != probe.GetNumAtoms() for m in maps):
        raise ValueError("Incomplete or capped chirality-preserving heavy-atom mapping enumeration")
    ref = np.asarray(reference.GetConformer().GetPositions())
    def rmsd(x):
        return min(float(np.sqrt(np.mean(np.sum((x - ref[list(m)]) ** 2, axis=1)))) for m in maps)
    ordered = ref[list(maps[0])]
    if abs(rmsd(ordered)) > 1e-12 or abs(rmsd(ordered + [10, 0, 0]) - 10) > 1e-10:
        raise ValueError("Fixed-frame RMSD translation control failed")
    scores = [float(l.split()[3]) for l in poses.read_text().splitlines() if l.startswith("REMARK VINA RESULT:")]
    if not scores or len(scores) != probe.GetNumConformers() or scores != sorted(scores):
        raise ValueError("Scores and ranked pose coordinates are inconsistent")
    rows = [{"rank": i + 1, "vina_score_kcal_mol": scores[i],
             "symmetry_aware_heavy_atom_rmsd_A": rmsd(np.asarray(c.GetPositions()))}
            for i, c in enumerate(probe.GetConformers())]
    return {"n_poses": len(rows), "graph_automorphisms": len(maps),
            "translation_control_10A_passed": True, "no_superposition": True,
            "top1_rmsd_A": rows[0]["symmetry_aware_heavy_atom_rmsd_A"],
            "best_top5_rmsd_A": min(r["symmetry_aware_heavy_atom_rmsd_A"] for r in rows[:5]),
            "first_recovery_rank": next((r["rank"] for r in rows if r["symmetry_aware_heavy_atom_rmsd_A"] <= 2), None),
            "computed_poses": rows}


def operation(command, log, timeout):
    start = time.monotonic()
    with log.open("x") as stream:
        try:
            p = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            result = {"command": command, "exit_code": p.returncode, "elapsed_seconds": time.monotonic() - start}
        except subprocess.TimeoutExpired:
            result = {"command": command, "exit_code": None, "timed_out": True,
                      "elapsed_seconds": time.monotonic() - start}
    write_json(log.with_suffix(".operation.json"), result)
    if result["exit_code"] != 0:
        raise RuntimeError(f"Operation failed; see {log.name}, exit={result['exit_code']}")


def run_target(row, root, sources, binary):
    destination = root / row["target"]
    receipt_file = destination / "receipt.json"
    if destination.exists():
        if not receipt_file.exists():
            raise RuntimeError(f"Incomplete previous target {destination}; preserve and inspect before retry")
        receipt = json.loads(receipt_file.read_text())
        for name, sha in receipt["output_sha256"].items():
            if digest(destination / name) != sha:
                raise ValueError("Existing target artifact changed")
        return receipt
    destination.mkdir()
    start = time.monotonic()
    receipt = {"target": row["target"], "pdb": row["pdb"], "ligand_code": row["ligand_code"],
               "published_reference": row, "fidelity": FIDELITY, "status": "started"}
    phase = "source"
    try:
        pdb = row["pdb"].lower()
        source = sources / pdb
        source.mkdir(parents=True, exist_ok=True)
        cif = source / f"{pdb}.cif"
        download(f"https://files.rcsb.org/download/{pdb.upper()}.cif", cif)
        structure = gemmi.read_structure(str(cif))
        structure.setup_entities()
        center = np.array([float(row[f"grid_center_{axis}_A"]) for axis in "xyz"])
        size = np.array([float(row[f"grid_size_{axis}_A"]) for axis in "xyz"])
        selected, candidates = select_ligand(structure, row["ligand_code"], center, size)
        receipt.update(native_selection=selected, candidates=candidates, deposited_models=len(structure))
        if selected["insertion_code"]:
            raise ValueError("Native ligand insertion code requires explicit instance API support")
        query = urllib.parse.urlencode({"auth_asym_id": selected["chain"], "auth_seq_id": selected["resnum"], "encoding": "sdf"})
        sdf = source / f"{selected['code']}_{selected['chain']}_{selected['resnum']}.sdf"
        download(f"https://models.rcsb.org/v1/{pdb}/ligand?{query}", sdf)
        receipt["source_sha256"] = {str(cif): digest(cif), str(sdf): digest(sdf)}
        phase = "native_coordinate_verification"
        receipt["ligand_verification"] = verify_native_ligand(sdf, selected, destination / "ligand_native_h.sdf")
        receipt["receptor_cleaning"] = receptor_pdb(structure, destination / "receptor_clean.pdb")
        bin_dir = Path(sys.executable).parent
        phase = "receptor_preparation"
        operation([str(bin_dir / "mk_prepare_receptor.py"), "--read_pdb", str(destination / "receptor_clean.pdb"),
                   "-p", str(destination / "receptor.pdbqt"), "--default_altloc", "A", "--charge_model", "gasteiger"],
                  destination / "receptor_preparation.log", 600)
        phase = "ligand_preparation"
        operation([str(bin_dir / "mk_prepare_ligand.py"), "-i", str(destination / "ligand_native_h.sdf"),
                   "-o", str(destination / "ligand.pdbqt")], destination / "ligand_preparation.log", 300)
        phase = "docking"
        command = [str(binary), "--receptor", str(destination / "receptor.pdbqt"),
                   "--ligand", str(destination / "ligand.pdbqt")]
        for name, values in [("center", center), ("size", size)]:
            for axis, value in zip("xyz", values):
                command += [f"--{name}_{axis}", str(value)]
        command += ["--scoring", "vina", "--exhaustiveness", "64", "--num_modes", "20", "--energy_range", "4",
                    "--seed", "42", "--cpu", "2", "--out", str(destination / "docked.pdbqt")]
        operation(command, destination / "vina.log", 1800)
        phase = "pose_verification"
        receipt["pose_recovery"] = pose_recovery(destination / "ligand_native_h.sdf", destination / "docked.pdbqt")
        receipt["status"] = "computed"
    except Exception as exc:
        receipt.update(status="failed", failure_phase=phase, error_type=type(exc).__name__, error=str(exc))
    receipt["elapsed_seconds"] = time.monotonic() - start
    receipt["output_sha256"] = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
    write_json(receipt_file, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--vina", type=Path, required=True)
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()
    actual = {name: version(name) for name in PINS}
    if actual != PINS or digest(args.vina) != VINA_SHA256:
        raise ValueError("Scientific environment or Vina binary differs from pinned reconstruction")
    if subprocess.check_output([str(args.vina), "--version"], text=True).strip() != "AutoDock Vina v1.2.5":
        raise ValueError("Require exact Vina 1.2.5 binary")
    rows = list(csv.DictReader(args.table.open()))
    eligible = [r for r in rows if r["evaluated"] == "True"]
    if args.targets:
        if set(args.targets) - {r["target"] for r in eligible}:
            raise ValueError("Requested target not in the published evaluated panel")
        eligible = [r for r in eligible if r["target"] in args.targets]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = {"protocol": "public_crystal_redocking_reconstruction.v1", "fidelity": FIDELITY,
            "table_sha256": digest(args.table), "worker_sha256": digest(__file__),
            "software": actual, "binary_sha256": digest(args.vina), "targets": [r["target"] for r in eligible],
            "author_exclusions": [r for r in rows if r["evaluated"] != "True"],
            "exhaustiveness": 64, "num_modes": 20, "energy_range": 4, "seed": 42, "cpu": 2}
    spec_file = root / "run_spec.json"
    if spec_file.exists() and json.loads(spec_file.read_text()) != spec:
        raise ValueError("Existing run has a different immutable specification")
    write_json(spec_file, spec)
    receipts = []
    for row in eligible:
        print(json.dumps({"target": row["target"], "phase": "starting"}), flush=True)
        receipt = run_target(row, root, args.sources.resolve(), args.vina.resolve())
        receipts.append(receipt)
        summary = {"requested": len(eligible), "finished": len(receipts),
                   "computed": sum(r["status"] == "computed" for r in receipts),
                   "failed": sum(r["status"] == "failed" for r in receipts), "targets": receipts}
        write_json(root / "summary.json", summary)
        recovery = receipt.get("pose_recovery", {})
        print(json.dumps({"target": row["target"], "status": receipt["status"],
                          "failure_phase": receipt.get("failure_phase"),
                          "best_top5_rmsd_A": recovery.get("best_top5_rmsd_A"),
                          "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
