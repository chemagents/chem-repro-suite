"""Validate deposited 3-D catalysts without conflating annotations and geometry."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from ..common.axial import inspect_axial_geometry
from ..common.chemistry import canonical_smiles
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, table
from .analysis import _metric, _result, load


def achiral_identity(mol):
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    # Remove stereo first: default RemoveHs otherwise retains an H defining imine E/Z,
    # which would make a representation difference look like a different heavy-atom graph.
    copy = Chem.RemoveHs(copy)
    return Chem.MolToSmiles(copy, isomericSmiles=True)


def compare_stereochemistry(expected, observed):
    """Extra centers in the SDF are not contradicted by unspecified source centers."""
    # Explicit H on both sides avoids deleting specified alkene/imine stereo while
    # making implicit and explicit H comparable. Isotopes and formal charges remain.
    expected, observed = Chem.Mol(expected), Chem.Mol(observed)
    # MMFF uses its own aromaticity model and may leave an in-memory molecule
    # kekulized. Re-sanitize copies before a stereo-sensitive graph comparison.
    Chem.SanitizeMol(expected)
    Chem.SanitizeMol(observed)
    expected, observed = Chem.AddHs(expected), Chem.AddHs(observed)
    same_graph = achiral_identity(expected) == achiral_identity(observed)
    specified_match = (
        same_graph
        and expected.GetNumAtoms() == observed.GetNumAtoms()
        and observed.HasSubstructMatch(expected, useChirality=True)
    )
    return {
        "connectivity_match": same_graph,
        "specified_source_stereo_match": bool(specified_match),
        "exact_isomeric_smiles_match": Chem.MolToSmiles(expected) == Chem.MolToSmiles(observed),
        "source_assigned_atom_centers": len(Chem.FindMolChiralCenters(expected, includeUnassigned=False)),
        "sdf_assigned_atom_centers": len(Chem.FindMolChiralCenters(observed, includeUnassigned=False)),
    }


def audit_catalyst_structures():
    df = load()
    references = {}
    for row in df.itertuples():
        name = Path(row.organocatalyst_3d_sdf).name
        identity = (canonical_smiles(row.organocatalyst), row.axial_configuration)
        if name in references and references[name] != identity:
            raise ValueError(f"One SDF filename maps to conflicting catalyst identities: {name}")
        references[name] = identity
    records = []
    with zipfile.ZipFile(DATA_ROOT / "mannich/organocatalysts_sdfs.zip") as z:
        files = {}
        metadata_count = 0
        for entry in z.infolist():
            if entry.filename.startswith("__MACOSX/") or Path(entry.filename).name.startswith("._"):
                metadata_count += 1
                continue
            if not entry.filename.lower().endswith(".sdf"):
                continue
            key = Path(entry.filename).name
            if key in files:
                raise ValueError(f"Duplicate SDF basename in archive: {key}")
            files[key] = entry.filename
        for name, (smiles, axial) in sorted(references.items()):
            rec = {
                "filename": name,
                "source_smiles": smiles,
                "source_axial_configuration": axial,
                "present": name in files,
                "parsed": False,
                "finite_3d_coordinates": False,
                "source_smiles_property_match": None,
                "axial_property_match": None,
                "connectivity_match": None,
                "specified_source_stereo_match": None,
                "exact_isomeric_smiles_match": None,
                "source_assigned_atom_centers": None,
                "sdf_assigned_atom_centers": None,
                "geometry_specified_stereo_match": None,
                "axial_bond_cip_labels": [],
                "geometry_axis_analysis": None,
                "axial_absolute_validation": "unverified_missing_structure"
                if axial in {"R", "S"}
                else "not_applicable",
            }
            if name in files:
                supplier = Chem.ForwardSDMolSupplier(io.BytesIO(z.read(files[name])), removeHs=False)
                mol = next(supplier, None)
                if mol is not None:
                    if next(supplier, None) is not None:
                        raise ValueError("Expected one catalyst per SDF")
                    rec["parsed"] = True
                    rec["finite_3d_coordinates"] = bool(
                        mol.GetNumConformers()
                        and mol.GetConformer().Is3D()
                        and np.isfinite(mol.GetConformer().GetPositions()).all()
                    )
                    rec["source_smiles_property_match"] = (
                        canonical_smiles(mol.GetProp("source_smiles")) == smiles
                        if mol.HasProp("source_smiles")
                        else None
                    )
                    rec["axial_property_match"] = (
                        mol.GetProp("axial_configuration").strip() == axial
                        if mol.HasProp("axial_configuration")
                        else None
                    )
                    expected = Chem.MolFromSmiles(smiles)
                    rec.update(compare_stereochemistry(expected, mol))
                    if rec["finite_3d_coordinates"]:
                        geom = Chem.Mol(mol)
                        Chem.AssignStereochemistryFrom3D(geom, replaceExistingTags=True)
                        rec["geometry_specified_stereo_match"] = compare_stereochemistry(expected, geom)[
                            "specified_source_stereo_match"
                        ]
                        if axial in {"R", "S"}:
                            inspected = inspect_axial_geometry(geom)
                            rec["geometry_axis_analysis"] = inspected
                            rec["axial_bond_cip_labels"] = [
                                a["cip_descriptor"]
                                for a in inspected["all_candidate_axes"]
                                if a["cip_descriptor"] is not None
                            ]
                            selected = inspected["selected_axis"]
                            rec["axial_absolute_validation"] = (
                                "unverified_" + inspected["selection_policy"]
                                if selected is None
                                else selected["status"]
                                if selected["status"] != "assigned_on_nominated_axis"
                                else "matches"
                                if selected["axial_rs"] == axial
                                else "disagrees"
                            )
                    elif axial in {"R", "S"}:
                        rec["axial_absolute_validation"] = "unverified_missing_geometry"
            records.append(rec)
    counts = {
        "expected_catalysts": len(references),
        "sdf_records": len(files),
        "ignored_archive_metadata": metadata_count,
        "parse_failures": sum(not r["parsed"] for r in records),
        "connectivity_disagreements": sum(r["connectivity_match"] is False for r in records),
        "specified_stereo_disagreements": sum(r["specified_source_stereo_match"] is False for r in records),
        "geometry_specified_stereo_disagreements": sum(
            r["geometry_specified_stereo_match"] is False for r in records
        ),
        "exact_isomeric_smiles_disagreements": sum(
            r["exact_isomeric_smiles_match"] is False for r in records
        ),
        "axial_annotations": sum(r["source_axial_configuration"] in {"R", "S"} for r in records),
        "axial_absolute_matches": sum(r["axial_absolute_validation"] == "matches" for r in records),
        "axial_geometry_disagreements": sum(r["axial_absolute_validation"] == "disagrees" for r in records),
        "axial_unresolved": sum(
            r["source_axial_configuration"] in {"R", "S"}
            and r["axial_absolute_validation"] not in {"matches", "disagrees"}
            for r in records
        ),
    }
    return _result(
        "Deposited catalyst structures were checked against explicit source identities and 3-D geometry. "
        "An R/S property or dihedral-sign convention alone is not an independent absolute axial-CIP assignment.",
        [_metric(k, v, population="unique catalyst SDF records") for k, v in counts.items()],
        {
            "catalyst_structure_audit": table(
                pd.DataFrame(records), "One row per actual SDF, excluding macOS resource forks"
            )
        },
        protocol="mannich.stereochemistry_audit.v3",
        hydrogen_policy="explicit H on both sides for stereo comparison; remove stereo before H for achiral identity",
        axial_policy="geometry-derived CW/CCW tag then RDKit CIP; unique ring-constrained axis, else unique distinguishable axis",
        axial_descriptor_mapping={"M": "R", "P": "S"},
        axial_nomenclature_source="https://iupac.qmul.ac.uk/BlueBook/P9.html#P-92.1.2.2.3",
        notes=[
            EvidenceNote(
                code="HYDROGEN_REPRESENTATION",
                severity="info",
                message="The earlier v1 graph flag for catalyst108.sdf was caused by an explicit "
                "hydrogen retained to define imine stereo. v2 normalizes H representation without "
                "rewriting source structures; isotope/charge and specified stereo differences remain detectable.",
            ),
            EvidenceNote(
                code="AXIAL_ASSIGNMENT_SCOPE",
                message='The deposited generator chooses "R" versus "S" using an atom-order-dependent '
                "torsion sign. v3 instead derives orientation from 3D and applies RDKit CIP sequence rules, "
                "with axial M=R and P=S (IUPAC P-92.1.2.2.3). A unique ring-constrained axis is preferred; "
                "multiple alternatives remain unresolved. The nominated axis must still be checked against "
                "the original drawing. Axes in rings smaller than 8 atoms remain conditional because of "
                "RDKit's support boundary. No rotational barrier or configurational lifetime was computed.",
            ),
            EvidenceNote(
                code="MANUAL_STEREOCHEMISTRY",
                message="Agreement with tabulated annotations does "
                "not replace manual comparison against the original reaction drawing. No OCR is performed.",
            ),
        ],
    )
