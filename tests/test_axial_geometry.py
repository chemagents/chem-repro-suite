import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

from chem_repro.common.axial import AXIAL_CIP_TO_RS, biaryl_bonds, geometry_axial_descriptor


def model_atropisomer(tag):
    mol = Chem.AddHs(Chem.MolFromSmiles("Cc1cccc(I)c1-c1c(Br)cccc1O"))
    axis = biaryl_bonds(mol)[0]
    bond = mol.GetBondWithIdx(axis)
    bond.SetStereo(tag)
    rdCIPLabeler.AssignCIPLabels(mol)
    label = bond.GetProp("_CIPCode")
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mol, axis, label


@pytest.mark.parametrize("tag", [Chem.BondStereo.STEREOATROPCW, Chem.BondStereo.STEREOATROPCCW])
def test_geometry_agrees_with_native_roundtrip_and_ignores_wrong_stored_tag(tag):
    mol, axis, native = model_atropisomer(tag)
    # Deliberately contradict the geometry. A metadata lookup would get this wrong.
    mol.GetBondWithIdx(axis).SetStereo(
        Chem.BondStereo.STEREOATROPCCW
        if tag == Chem.BondStereo.STEREOATROPCW
        else Chem.BondStereo.STEREOATROPCW
    )
    result = geometry_axial_descriptor(mol, axis)
    assert result["cip_descriptor"] == native
    assert result["axial_rs"] == AXIAL_CIP_TO_RS[native]
    assert result["status"] == "assigned_on_nominated_axis"
    # In 3D native MolBlock perception uses coordinates, with wedges only marking the axis.
    restored = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol, forceV3000=True), removeHs=False)
    rdCIPLabeler.AssignCIPLabels(restored)
    a, b = mol.GetBondWithIdx(axis).GetBeginAtomIdx(), mol.GetBondWithIdx(axis).GetEndAtomIdx()
    assert restored.GetBondBetweenAtoms(a, b).GetProp("_CIPCode") == native


def test_atom_permutation_rotation_and_mirror_controls():
    mol, axis, label = model_atropisomer(Chem.BondStereo.STEREOATROPCW)
    rng = np.random.default_rng(73)
    a, b = mol.GetBondWithIdx(axis).GetBeginAtomIdx(), mol.GetBondWithIdx(axis).GetEndAtomIdx()
    order = rng.permutation(mol.GetNumAtoms()).tolist()
    renumbered = Chem.RenumberAtoms(mol, order)
    remapped = renumbered.GetBondBetweenAtoms(order.index(a), order.index(b)).GetIdx()
    assert geometry_axial_descriptor(renumbered, remapped)["cip_descriptor"] == label
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    original = mol.GetConformer().GetPositions()
    for i, point in enumerate(original @ q + [12, -4, 7]):
        mol.GetConformer().SetAtomPosition(i, point)
    assert geometry_axial_descriptor(mol, axis)["cip_descriptor"] == label
    for i, point in enumerate(original * [-1, 1, 1]):
        mol.GetConformer().SetAtomPosition(i, point)
    assert geometry_axial_descriptor(mol, axis)["cip_descriptor"] == {"M": "P", "P": "M"}[label]


def test_planar_coordinates_do_not_prove_axial_configuration():
    mol, axis, _ = model_atropisomer(Chem.BondStereo.STEREOATROPCW)
    xyz = mol.GetConformer().GetPositions()
    for i, point in enumerate(xyz * [1, 1, 0]):
        mol.GetConformer().SetAtomPosition(i, point)
    result = geometry_axial_descriptor(mol, axis)
    assert result["axial_rs"] is None
