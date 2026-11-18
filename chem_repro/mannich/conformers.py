"""Reproduce source catalyst generation, with an explicitly separate CIP-checked alternative."""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler, rdMolAlign

from ..common.artifacts import save_artifact
from ..common.axial import inspect_axial_geometry
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, table
from ..vendor.mannich_source_generator import _mmff_energy, build_3d_catalyst
from .analysis import _metric, _result, load
from .stereochemistry import compare_stereochemistry


def _single_conformer(mol, conf_id):
    copied = Chem.Mol(mol)
    copied.RemoveAllConformers()
    copied.AddConformer(mol.GetConformer(conf_id), assignId=True)
    return copied


def generate_one(smiles, axial, *, protocol="source_notebook", seed=42):
    if protocol not in {"source_notebook", "geometry_verified"}:
        raise ValueError("Unknown generation protocol")
    mol, status, info = build_3d_catalyst(smiles, axial, random_seed=seed)
    diagnostics = {"source_status": status, "source_diagnostics": info, "candidate_diagnostics": []}
    if mol is None:
        return None, status, diagnostics
    if protocol == "source_notebook" or axial not in {"R", "S"}:
        return mol, status, diagnostics
    nominated = inspect_axial_geometry(mol)
    diagnostics["axis_nomination"] = nominated
    axis = nominated["selected_axis"]
    if axis is None or axis["status"] != "assigned_on_nominated_axis":
        return None, "unsupported_axial_nomination", diagnostics
    # Source and returned molecule preserve atom order. Assert connectivity rather than
    # transferring an atom index blindly if a future source function changes this behavior.
    source = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if source.GetNumAtoms() != mol.GetNumAtoms() or any(
        source.GetAtomWithIdx(i).GetAtomicNum() != mol.GetAtomWithIdx(i).GetAtomicNum()
        for i in range(source.GetNumAtoms())
    ):
        return None, "atom_order_changed", diagnostics
    if source.GetNumBonds() != mol.GetNumBonds() or any(
        mol.GetBondBetweenAtoms(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) is None for b in source.GetBonds()
    ):
        return None, "atom_connectivity_order_changed", diagnostics
    a, b = axis["axis_atoms"]
    bond = source.GetBondBetweenAtoms(a, b)
    if bond is None:
        return None, "source_axis_missing", diagnostics
    expected_cip = {"R": "M", "S": "P"}[axial]
    for tag in [Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW]:
        if bond.HasProp("_CIPCode"):
            bond.ClearProp("_CIPCode")
        bond.SetStereo(tag)
        rdCIPLabeler.AssignCIPLabels(source, bondsToLabel=[bond.GetIdx()], maxRecursiveIterations=1000000)
        if bond.HasProp("_CIPCode") and bond.GetProp("_CIPCode") == expected_cip:
            break
    else:
        return None, "cannot_encode_requested_axial_configuration", diagnostics
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.useSmallRingTorsions = True
    parameters.enforceChirality = True
    parameters.numThreads = 1
    conformers = list(AllChem.EmbedMultipleConfs(source, numConfs=5, params=parameters))
    if not conformers:
        parameters.useRandomCoords = True
        parameters.randomSeed = seed + 100
        conformers = list(AllChem.EmbedMultipleConfs(source, numConfs=5, params=parameters))
    if not conformers:
        return None, "verified_embedding_failed", diagnostics
    candidates = []
    for cid in conformers:
        optimization = int(AllChem.MMFFOptimizeMolecule(source, confId=cid, maxIters=500))
        generated = _single_conformer(source, cid)
        Chem.AssignStereochemistryFrom3D(generated, replaceExistingTags=True)
        chemical_match = compare_stereochemistry(Chem.MolFromSmiles(smiles), generated)
        assigned = inspect_axial_geometry(generated)["selected_axis"]
        energy = float(_mmff_energy(generated))
        valid = bool(
            chemical_match["specified_source_stereo_match"]
            and assigned is not None
            and assigned["status"] == "assigned_on_nominated_axis"
            and assigned["axial_rs"] == axial
            and np.isfinite(energy)
        )
        diagnostics["candidate_diagnostics"].append(
            {
                "conformer": int(cid),
                "optimization_status": optimization,
                "mmff_energy_kcal_mol": energy if np.isfinite(energy) else None,
                "selected_axis": assigned,
                "source_point_stereo_matches": chemical_match["specified_source_stereo_match"],
                "accepted": valid,
            }
        )
        if valid:
            candidates.append((energy, generated, optimization))
    if not candidates:
        return None, "no_verified_axial_conformer", diagnostics
    candidates.sort(key=lambda item: item[0])
    diagnostics["selected_optimization_status"] = candidates[0][2]
    return candidates[0][1], "geometry_verified_axial", diagnostics


def regenerate_catalysts(protocol="source_notebook", filenames=None, seed=42, progress=None):
    if protocol not in {"source_notebook", "geometry_verified"}:
        raise ValueError("Unknown generation protocol")
    df = load()
    refs = df.drop_duplicates("organocatalyst_3d_sdf").copy()
    refs["source_filename"] = refs.organocatalyst_3d_sdf.map(lambda v: Path(v).name)
    if filenames is not None:
        if (
            not filenames
            or len(set(filenames)) != len(filenames)
            or set(filenames) - set(refs.source_filename)
        ):
            raise ValueError("Require unique filenames from the pinned catalyst registry")
        refs = refs.loc[refs.source_filename.isin(filenames)]
    refs = refs.sort_values("source_filename")
    rows, body = [], io.BytesIO()
    with (
        zipfile.ZipFile(DATA_ROOT / "mannich/organocatalysts_sdfs.zip") as originals,
        zipfile.ZipFile(body, "w", zipfile.ZIP_DEFLATED) as generated_zip,
    ):
        original_files = {
            Path(n).name: n
            for n in originals.namelist()
            if n.endswith(".sdf") and not n.startswith("__MACOSX/")
        }
        for entry in refs.itertuples():
            filename, axial = entry.source_filename, entry.axial_configuration
            mol, status, diagnostics = generate_one(entry.organocatalyst, axial, protocol=protocol, seed=seed)
            row = {
                "filename": filename,
                "source_smiles": entry.organocatalyst,
                "source_axial_configuration": axial,
                "generation_status": status,
                "generated": mol is not None,
                "finite_3d_coordinates": False,
                "graph_matches": None,
                "specified_point_stereo_matches": None,
                "mmff_energy_kcal_mol": None,
                "aligned_heavy_atom_rmsd_to_deposit_A": None,
                "axial_geometry": None,
                "axial_annotation_comparison": "unresolved" if axial in {"R", "S"} else "not_applicable",
                "source_axis_matches_nominated": None,
                "diagnostics": diagnostics,
            }
            if mol is not None:
                row["finite_3d_coordinates"] = bool(
                    mol.GetConformer().Is3D() and np.isfinite(mol.GetConformer().GetPositions()).all()
                )
                checks = compare_stereochemistry(Chem.MolFromSmiles(entry.organocatalyst), mol)
                row["graph_matches"] = checks["connectivity_match"]
                row["specified_point_stereo_matches"] = checks["specified_source_stereo_match"]
                energy = float(_mmff_energy(mol))
                row["mmff_energy_kcal_mol"] = energy if np.isfinite(energy) else None
                if axial in {"R", "S"}:
                    row["axial_geometry"] = inspect_axial_geometry(mol)
                    chosen = row["axial_geometry"]["selected_axis"]
                    if chosen is not None:
                        prefix = "" if chosen["status"] == "assigned_on_nominated_axis" else "conditional_"
                        row["axial_annotation_comparison"] = prefix + (
                            "matches" if chosen["axial_rs"] == axial else "disagrees"
                        )
                        source_axis = diagnostics["source_diagnostics"].get("axis")
                        row["source_axis_matches_nominated"] = (
                            set(source_axis) == set(chosen["axis_atoms"]) if source_axis else None
                        )
                deposited = next(
                    Chem.ForwardSDMolSupplier(
                        io.BytesIO(originals.read(original_files[filename])), removeHs=False
                    )
                )
                if deposited is not None:
                    try:
                        row["aligned_heavy_atom_rmsd_to_deposit_A"] = float(
                            rdMolAlign.GetBestRMS(
                                Chem.RemoveAllHs(mol), Chem.RemoveAllHs(deposited), maxMatches=1000
                            )
                        )
                    except RuntimeError:
                        row["diagnostics"]["deposit_rmsd_error"] = "no_valid_heavy_atom_alignment"
                mol.SetProp("source_smiles", entry.organocatalyst)
                mol.SetProp("axial_configuration", axial)
                mol.SetProp("generation_protocol", protocol)
                mol.SetProp("generation_status", status)
                mol.SetProp(
                    "axis_validation_scope",
                    "nominated axis; geometry and CIP, no rotational-barrier inference",
                )
                mol.SetProp("_Name", filename.removesuffix(".sdf"))
                buffer = io.StringIO()
                writer = Chem.SDWriter(buffer)
                writer.write(mol)
                writer.close()
                generated_zip.writestr(filename, buffer.getvalue())
            rows.append(row)
            if progress:
                progress(row)
        generated_zip.writestr("generation_records.json", json.dumps(rows, indent=2, allow_nan=False))
    n_generated = sum(r["generated"] for r in rows)
    contradictions = sum(r["specified_point_stereo_matches"] is False for r in rows)
    axial_rows = [r for r in rows if r["source_axial_configuration"] in {"R", "S"}]
    axial_matches = sum(
        bool(
            (r["axial_geometry"] or {}).get("selected_axis")
            and r["axial_geometry"]["selected_axis"]["status"] == "assigned_on_nominated_axis"
            and r["axial_geometry"]["selected_axis"]["axial_rs"] == r["source_axial_configuration"]
        )
        for r in axial_rows
    )
    result = _result(
        f"Generated {n_generated}/{len(rows)} requested catalyst structures using {protocol}. "
        f"{contradictions} contradict source-specified point stereochemistry; {axial_matches}/{len(axial_rows)} "
        "axial annotations match geometry-derived CIP on the nominated supported axis. "
        "Source and geometry-verified protocols are separate outputs; source files are not overwritten.",
        [
            _metric("requested_catalysts", len(rows)),
            _metric("generated_catalysts", n_generated),
            _metric("failed_generation", len(rows) - n_generated),
            _metric("point_stereo_contradictions", contradictions),
            _metric("axial_annotations", len(axial_rows)),
            _metric("axial_geometry_matches", axial_matches),
            _metric(
                "axial_geometry_disagreements",
                sum(r["axial_annotation_comparison"] == "disagrees" for r in rows),
            ),
            _metric(
                "axial_conditional_matches",
                sum(r["axial_annotation_comparison"] == "conditional_matches" for r in rows),
            ),
            _metric(
                "axial_conditional_disagreements",
                sum(r["axial_annotation_comparison"] == "conditional_disagrees" for r in rows),
            ),
            _metric("source_axis_mismatches", sum(r["source_axis_matches_nominated"] is False for r in rows)),
        ],
        {
            "generation_records": table(
                pd.DataFrame(rows), "Every requested catalyst, including failures and unresolved axes"
            )
        },
        protocol="mannich.regeneration." + protocol + ".v1",
        status="partial",
        seed=seed,
        conformers=5,
        bridged_seeds=20,
        mmff_max_iterations=500,
        source_algorithm_revision="9cbd97285d945c3131d460f7f1ad33b9e0be76ef",
        rmsd_scope="symmetry-aware rigidly aligned heavy-atom RMSD to deposited SDF; maxMatches=1000",
        notes=[
            EvidenceNote(
                code="GENERATION_PROTOCOLS",
                message="source_notebook retains the exact source "
                "ETKDGv3/MMFF94 and torsion-sign algorithm. Its sign rule depends on atom order and is not "
                "absolute CIP. geometry_verified is an explicit alternative: encode requested CIP before "
                "embedding, optimize, then independently check geometry and filter invalid candidates. "
                "Ambiguous axes and native-unsupported small rings fail rather than silently acquire labels.",
            ),
            EvidenceNote(
                code="CONFORMER_SCOPE",
                message="Selected energy is relative to sampled conformers, "
                "not a proven global minimum. Original functions do not report every optimization convergence "
                "code. Finite energy is not proof of convergence. Version-dependent coordinates can differ "
                "from deposited SDFs; this RMSD is not a docking validation score.",
            ),
        ],
    )
    result.artifacts.append(
        save_artifact(
            body.getvalue(),
            suffix="zip",
            media_type="application/zip",
            public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7343"),
            display_name="mannich_regenerated_" + protocol + ".zip",
        )
    )
    return result
