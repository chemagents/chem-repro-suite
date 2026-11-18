from rdkit import Chem

from chem_repro.common.axial import inspect_axial_geometry
from chem_repro.mannich.conformers import generate_one
from chem_repro.mannich.stereochemistry import compare_stereochemistry


def test_source_generation_preserves_specified_point_stereo():
    smiles = "N[C@H](C)C(=O)O"
    mol, status, info = generate_one(smiles, "none")
    assert status == "smiles"
    assert mol is not None and mol.GetConformer().Is3D()
    assert compare_stereochemistry(Chem.MolFromSmiles(smiles), mol)["specified_source_stereo_match"]
    assert info["source_diagnostics"]["strategy"] == "plain"


def test_verified_generation_is_checked_against_geometry():
    smiles = "Cc1cccc(I)c1-c1c(Br)cccc1O"
    mol, status, info = generate_one(smiles, "R", protocol="geometry_verified")
    assert mol is not None and status == "geometry_verified_axial"
    assert inspect_axial_geometry(mol)["selected_axis"]["axial_rs"] == "R"
    assert any(c["accepted"] for c in info["candidate_diagnostics"])
