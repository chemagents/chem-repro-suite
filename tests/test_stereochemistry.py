from rdkit import Chem

from chem_repro.mannich.stereochemistry import audit_catalyst_structures, compare_stereochemistry


def test_opposite_defined_center_is_a_real_conflict():
    left = Chem.MolFromSmiles("N[C@H](C)C(=O)O")
    right = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    r = compare_stereochemistry(left, right)
    assert r["connectivity_match"]
    assert not r["specified_source_stereo_match"]


def test_extra_center_is_distinct_from_contradicted_center():
    source = Chem.MolFromSmiles("NC(C)C(=O)O")
    observed = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    r = compare_stereochemistry(source, observed)
    assert r["specified_source_stereo_match"]
    assert not r["exact_isomeric_smiles_match"]


def test_kekule_and_aromatic_representations_do_not_create_stereo_conflicts():
    source = Chem.MolFromSmiles("O=c1c(N[C@@H](C)CO)c(N[C@@H](C)CO)c1=O")
    observed = Chem.Mol(source)
    Chem.Kekulize(observed, clearAromaticFlags=True)
    r = compare_stereochemistry(source, observed)
    assert r["connectivity_match"] and r["specified_source_stereo_match"] and r["exact_isomeric_smiles_match"]


def test_all_real_sdfs_audited_once():
    r = audit_catalyst_structures()
    m = {v.name: v.value for v in r.answer.metrics}
    assert m["expected_catalysts"] == m["sdf_records"] == 468
    assert m["parse_failures"] == 0
    rows = r.answer.tables["catalyst_structure_audit"].rows
    assert len({v["filename"] for v in rows}) == 468
    assert all(not v["filename"].startswith("._") for v in rows)
    flagged = next(v for v in rows if v["filename"] == "catalyst108.sdf")
    assert flagged["connectivity_match"] and flagged["specified_source_stereo_match"]
    assert m["connectivity_disagreements"] == 0


def test_explicit_imine_hydrogen_is_not_a_graph_conflict():
    source = Chem.MolFromSmiles("N=C(N)Nc1ccccc1")
    observed = Chem.MolFromSmiles("[H]/N=C(/N)Nc1ccccc1")
    r = compare_stereochemistry(source, observed)
    assert r["connectivity_match"] and r["specified_source_stereo_match"]
    assert not r["exact_isomeric_smiles_match"]


def test_defined_double_bond_stereo_and_isotopes_survive_comparison():
    trans = Chem.MolFromSmiles("F/C=C/F")
    cis = Chem.MolFromSmiles("F/C=C\\F")
    assert not compare_stereochemistry(trans, cis)["specified_source_stereo_match"]
    assert not compare_stereochemistry(Chem.MolFromSmiles("[2H]O"), Chem.MolFromSmiles("O"))[
        "connectivity_match"
    ]
