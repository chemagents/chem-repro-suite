"""Portable structural and reported-ratio diagnostics without implicit data repair."""

import re
from collections import Counter
from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=32768)
def structure_diagnostics(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "parsed": False,
            "heavy_atom_inventory": {},
            "formal_charge": None,
            "fragments": None,
            "unassigned_carbon_stereocenters": [],
        }
    inventory = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            name = (str(atom.GetIsotope()) if atom.GetIsotope() else "") + atom.GetSymbol()
            inventory[name] += 1
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {
        "parsed": True,
        "heavy_atom_inventory": dict(sorted(inventory.items())),
        "formal_charge": Chem.GetFormalCharge(mol),
        "fragments": len(Chem.GetMolFrags(mol)),
        "unassigned_carbon_stereocenters": [
            i for i, label in centers if label == "?" and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
        ],
    }


def heavy_atom_balance(reactants, product):
    """Product minus one copy of each recorded reactant. Not a full reaction-equation validation."""
    inputs = [structure_diagnostics(s) for s in reactants]
    output = structure_diagnostics(product)
    if not all(r["parsed"] for r in [*inputs, output]):
        raise ValueError("Cannot compare atom inventories of invalid structures")
    atoms = set(output["heavy_atom_inventory"]).union(*(r["heavy_atom_inventory"] for r in inputs))
    delta = {
        a: output["heavy_atom_inventory"].get(a, 0) - sum(r["heavy_atom_inventory"].get(a, 0) for r in inputs)
        for a in sorted(atoms)
    }
    return {
        "heavy_atom_delta": {a: v for a, v in delta.items() if v},
        "formal_charge_delta": output["formal_charge"] - sum(r["formal_charge"] for r in inputs),
    }


def parse_reported_ratio(value):
    """Keep a censoring relation and every ratio term; never reinterpret a dr as ee."""
    text = str(value).strip()
    match = re.fullmatch(r"(>=|<=|>|<|≥|≤)?\s*(\d+(?:[.,]\d+)?(?:\s*:\s*\d+(?:[.,]\d+)?)+)", text)
    if not match:
        raise ValueError("Expected a colon-delimited nonnegative ratio, optionally censored")
    relation = match.group(1) or "="
    terms = [float(v.strip().replace(",", ".")) for v in match.group(2).split(":")]
    if sum(terms) <= 0:
        raise ValueError("A ratio cannot have only zero terms")
    return {
        "relation": relation,
        "terms": terms,
        "reported_fractions": [v / sum(terms) for v in terms] if relation == "=" else None,
    }
