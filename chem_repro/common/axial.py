"""Geometry-based biaryl descriptors with explicit scope and axis selection.

The orientation convention follows RDKit Release_2026_03_3 Atropisomers.cpp:
lowest-index adjacent atoms define the internal CW/CCW tag; rdCIPLabeler then
applies the CIP sequence rules. Source R/S annotations do not enter assignment.
For axial units, M corresponds to Ra and P to Sa under IUPAC Blue Book
P-92.1.2.2.3. This conversion must not be used for planar chirality.
No rotational barrier or configurational lifetime is inferred from a conformer.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

ATROP_TAGS = {Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW}
AXIAL_CIP_TO_RS = {"M": "R", "P": "S"}


def biaryl_bonds(mol):
    """Aromatic-carbon single bonds connecting distinct aromatic ring systems."""
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    aromatic_rings = [
        set(r) for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    ]
    return [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE
        and not b.GetIsAromatic()
        and all(a.GetAtomicNum() == 6 and a.GetIsAromatic() for a in [b.GetBeginAtom(), b.GetEndAtom()])
        and not any({b.GetBeginAtomIdx(), b.GetEndAtomIdx()}.issubset(r) for r in aromatic_rings)
    ]


def geometry_axial_descriptor(mol, bond_index):
    """Assign one nominated biaryl axis from coordinates, never from a stored label."""
    if bond_index not in biaryl_bonds(mol):
        raise ValueError("Nominated bond is not a biaryl aromatic-carbon axis")
    result = {
        "bond_index": bond_index,
        "axis_atoms": None,
        "reference_atoms": None,
        "signed_sine": None,
        "internal_stereo_tag": None,
        "cip_descriptor": None,
        "axial_rs": None,
        "status": "unverified_geometry",
        "minimum_axis_ring_size": None,
    }
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    for b in work.GetBonds():
        if b.GetStereo() in ATROP_TAGS:
            b.SetStereo(Chem.BondStereo.STEREONONE)
        if b.HasProp("_CIPCode"):
            b.ClearProp("_CIPCode")
    bond = work.GetBondWithIdx(bond_index)
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    result["axis_atoms"] = [a, b]
    rings = [len(r) for r in work.GetRingInfo().BondRings() if bond_index in r]
    result["minimum_axis_ring_size"] = min(rings) if rings else None
    if not work.GetNumConformers() or not work.GetConformer().Is3D():
        return result
    xyz = work.GetConformer().GetPositions()
    if not np.isfinite(xyz).all():
        return result
    axis = xyz[b] - xyz[a]
    if np.linalg.norm(axis) < 1e-7:
        return result
    axis = axis / np.linalg.norm(axis)
    refs, projected = [], []
    for atom, partner in [(a, b), (b, a)]:
        neighbors = sorted(
            n.GetIdx() for n in work.GetAtomWithIdx(atom).GetNeighbors() if n.GetIdx() != partner
        )
        if len(neighbors) != 2:
            result["status"] = "unsupported_axis_valence"
            return result
        vectors = [xyz[n] - xyz[atom] for n in neighbors]
        vectors = [v - np.dot(v, axis) * axis for v in vectors]
        if np.linalg.norm(vectors[0]) < 1e-7 or np.dot(vectors[0], vectors[1]) > 1e-7:
            result["status"] = "unsupported_neighbor_geometry"
            return result
        refs.append(neighbors[0])
        projected.append(vectors[0] / np.linalg.norm(vectors[0]))
    signed = float(np.dot(np.cross(projected[0], projected[1]), axis))
    result["reference_atoms"], result["signed_sine"] = refs, signed
    if abs(signed) < 1e-7:
        result["status"] = "near_coplanar_axis"
        return result
    tag = Chem.BondStereo.STEREOATROPCW if signed > 0 else Chem.BondStereo.STEREOATROPCCW
    bond.SetStereo(tag)
    try:
        # Point configurations must also come from the conformer, not stale atom tags.
        Chem.AssignStereochemistryFrom3D(work, replaceExistingTags=True)
        bond.SetStereo(tag)
        rdCIPLabeler.AssignCIPLabels(work, bondsToLabel=[bond_index], maxRecursiveIterations=1000000)
    except RuntimeError:
        result["status"] = "cip_assignment_failed"
        return result
    label = bond.GetProp("_CIPCode") if bond.HasProp("_CIPCode") else None
    result.update(internal_stereo_tag=str(tag), cip_descriptor=label, axial_rs=AXIAL_CIP_TO_RS.get(label))
    if label not in AXIAL_CIP_TO_RS:
        result["status"] = "no_distinct_axial_CIP" if label is None else "unsupported_pseudoasymmetry"
    elif rings and min(rings) < 8:
        # RDKit's documented native atropisomer support excludes smaller ring axes.
        result["status"] = "conditional_assignment_small_ring"
    else:
        result["status"] = "assigned_on_nominated_axis"
    return result


def inspect_axial_geometry(mol):
    rows = [geometry_axial_descriptor(mol, i) for i in biaryl_bonds(mol)]
    assigned = [r for r in rows if r["axial_rs"] is not None]
    # The ring constraint is a structural selection rule, not a test of agreement
    # with source labels. Multiple possible ring axes remain explicitly unresolved.
    ring_axes = [r for r in assigned if r["minimum_axis_ring_size"] is not None]
    if len(ring_axes) == 1:
        selected, policy = ring_axes[0], "unique_ring_constrained_biaryl_axis"
    elif len(assigned) == 1:
        selected, policy = assigned[0], "unique_CIP_distinguishable_biaryl_axis"
    else:
        selected, policy = None, "ambiguous_axes" if assigned else "no_assignable_axis"
    return {"selection_policy": policy, "selected_axis": selected, "all_candidate_axes": rows}
