"""Functions extracted verbatim from ChemAgents Mannich source notebook.

MIT Copyright (c) 2026 ChemAgents; see Mannich_LICENSE.txt.
Source revision: 9cbd97285d945c3131d460f7f1ad33b9e0be76ef.
Only function definitions and their numeric constants are retained: no notebook
I/O, dependency installation, global warning suppression, or table rewriting.
The source torsion-sign convention is NOT an absolute axial CIP validation.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

RANDOM_SEED = 42
N_CONFORMERS = 5
N_BRIDGED_SEEDS = 20
MAX_OPT_ITERS = 500

def _is_filled(value) -> bool:
    """Return True if a cell value is meaningfully populated."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "na"):
        return False
    return True

def find_biaryl_axis(mol):
    """Locate a biaryl single bond suitable for an atropoisomeric axis.

    Returns a tuple ``(atom_idx_1, atom_idx_2)`` or ``None`` if no such bond
    is present. Both atoms must be aromatic carbons in distinct aromatic rings,
    and both ortho positions must be substituted (creating a rotation barrier).
    """
    ri = mol.GetRingInfo()
    aromatic_rings = [r for r in ri.AtomRings()
                      if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in r)]
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIsAromatic():
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if not (a1.GetIsAromatic() and a2.GetIsAromatic()):
            continue
        if a1.GetSymbol() != "C" or a2.GetSymbol() != "C":
            continue
        rings_a1 = {tuple(r) for r in aromatic_rings if a1.GetIdx() in r}
        rings_a2 = {tuple(r) for r in aromatic_rings if a2.GetIdx() in r}
        if rings_a1 & rings_a2 or not rings_a1 or not rings_a2:
            continue

        def _ortho_substituted(atom, partner_idx):
            for n in atom.GetNeighbors():
                if n.GetIdx() == partner_idx or not n.GetIsAromatic():
                    continue
                for sub in n.GetNeighbors():
                    if sub.GetIdx() in (atom.GetIdx(), partner_idx):
                        continue
                    if sub.GetSymbol() != "H":
                        return True
            return False

        if _ortho_substituted(a1, a2.GetIdx()) and _ortho_substituted(a2, a1.GetIdx()):
            return (a1.GetIdx(), a2.GetIdx())
    return None

def _ortho_neighbor(mol, atom_idx, partner_idx):
    """Return the index of an aromatic ortho neighbor of ``atom_idx``."""
    for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        if n.GetIdx() != partner_idx and n.GetIsAromatic():
            return n.GetIdx()
    return None

def _bond_in_ring(mol, idx1, idx2):
    """Return True if the bond between two atoms is part of a ring."""
    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    return bond.IsInRing() if bond is not None else False

def _measure_dihedral(mol, axis, conf_id=-1):
    """Measure the biaryl dihedral angle for a given axis."""
    a1, a2 = axis
    o1 = _ortho_neighbor(mol, a1, a2)
    o2 = _ortho_neighbor(mol, a2, a1)
    if o1 is None or o2 is None:
        return None
    conf = mol.GetConformer(conf_id) if conf_id >= 0 else mol.GetConformer()
    return rdMolTransforms.GetDihedralDeg(conf, o1, a1, a2, o2)

def _mmff_energy(mol, conf_id=-1):
    """Return MMFF94 energy of a conformer, or infinity on failure."""
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
        return ff.CalcEnergy() if ff else np.inf
    except Exception:
        return np.inf

def _embed(mol, seed, use_random_coords=False):
    """Embed a single conformer with ETKDGv3."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.enforceChirality = True
    params.useRandomCoords = use_random_coords
    return AllChem.EmbedMolecule(mol, params)

def build_3d_catalyst(smiles, axial_config="none",
                     n_conformers=N_CONFORMERS,
                     n_bridged_seeds=N_BRIDGED_SEEDS,
                     random_seed=RANDOM_SEED):
    """Generate a 3D conformer for an organocatalyst.

    The strategy is selected automatically:

    * ``plain`` — standard embedding for catalysts without a biaryl axis.
    * ``free_biaryl`` — embedding plus torsion constraint to enforce
      the requested axial configuration for free biaryls.
    * ``bridged_biaryl`` — multiple ETKDGv3 seeds for biaryl axes inside
      a ring, retaining conformers with the correct dihedral sign and
      returning the lowest-energy one.

    Returns
    -------
    mol : RDKit Mol or None
        Molecule with a single 3D conformer (with hydrogens), or None on failure.
    source : str
        Strategy label (``"smiles"``, ``"smiles+axial_free"``,
        ``"smiles+axial_bridged"``, ``"smiles_no_axis_applied"``)
        or an error code.
    info : dict
        Diagnostic data including ``axis``, ``strategy``, ``final_angle``,
        and ``n_valid_conformers`` (for the bridged strategy).
    """
    if not _is_filled(smiles):
        return None, "no_smiles", {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles", {}

    mol = Chem.AddHs(mol)

    axis = find_biaryl_axis(mol)
    needs_axial = _is_filled(axial_config) and str(axial_config).strip().upper() in ("R", "S")
    config = str(axial_config).strip().upper() if needs_axial else None

    if not needs_axial:
        strategy = "plain"
    elif axis is None:
        strategy = "plain_no_axis"
    else:
        strategy = "bridged_biaryl" if _bond_in_ring(mol, axis[0], axis[1]) else "free_biaryl"

    info = {"axis": axis, "strategy": strategy}

    # --- Strategy 1: bridged biaryl (NOBIN-type) ---
    if strategy == "bridged_biaryl":
        target_sign = +1 if config == "R" else -1
        candidates = []

        for seed in range(random_seed, random_seed + n_bridged_seeds):
            mol_try = Chem.AddHs(Chem.MolFromSmiles(smiles))
            rc = _embed(mol_try, seed)
            if rc < 0:
                rc = _embed(mol_try, seed, use_random_coords=True)
            if rc < 0:
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol_try, maxIters=MAX_OPT_ITERS)
            except Exception:
                continue

            angle = _measure_dihedral(mol_try, axis)
            if angle is None or np.sign(angle) != target_sign:
                continue
            candidates.append((_mmff_energy(mol_try), mol_try, angle))

        if not candidates:
            return None, f"bridged_no_{config}_conformer", info

        candidates.sort(key=lambda x: x[0])
        best_mol = candidates[0][1]
        info["final_angle"] = candidates[0][2]
        info["n_valid_conformers"] = len(candidates)
        Chem.AssignStereochemistryFrom3D(best_mol)
        return best_mol, "smiles+axial_bridged", info

    # --- Strategy 2: free biaryl (classical BINOL) ---
    if strategy == "free_biaryl":
        a1, a2 = axis
        o1 = _ortho_neighbor(mol, a1, a2)
        o2 = _ortho_neighbor(mol, a2, a1)

        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))

        if not conf_ids:
            for sd in range(random_seed + 100, random_seed + 110):
                rc = _embed(mol, sd, use_random_coords=True)
                if rc >= 0:
                    conf_ids = [rc]
                    break

        if not conf_ids:
            return None, "embed_failed", info

        target = 90.0 if config == "R" else -90.0
        for cid in conf_ids:
            try:
                rdMolTransforms.SetDihedralDeg(mol.GetConformer(cid), o1, a1, a2, o2, target)
                mp = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                if ff is not None:
                    lo, hi = (85.0, 95.0) if config == "R" else (-95.0, -85.0)
                    ff.MMFFAddTorsionConstraint(o1, a1, a2, o2, False, lo, hi, 1.0e4)
                    ff.Minimize(maxIts=MAX_OPT_ITERS)
                else:
                    AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=MAX_OPT_ITERS)
            except Exception:
                pass

        best_cid = min(conf_ids, key=lambda cid: _mmff_energy(mol, cid))
        best_mol = Chem.Mol(mol)
        best_mol.RemoveAllConformers()
        best_mol.AddConformer(mol.GetConformer(best_cid), assignId=True)
        Chem.AssignStereochemistryFrom3D(best_mol)
        info["final_angle"] = _measure_dihedral(best_mol, axis)
        return best_mol, "smiles+axial_free", info

    # --- Strategy 3: plain (point chirality from SMILES) ---
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useSmallRingTorsions = True
    params.enforceChirality = True
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))

    if not conf_ids:
        for sd in range(random_seed + 100, random_seed + 110):
            rc = _embed(mol, sd, use_random_coords=True)
            if rc >= 0:
                conf_ids = [rc]
                break

    if not conf_ids:
        return None, "embed_failed", info

    for cid in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=MAX_OPT_ITERS)
        except Exception:
            pass

    best_cid = min(conf_ids, key=lambda cid: _mmff_energy(mol, cid))
    best_mol = Chem.Mol(mol)
    best_mol.RemoveAllConformers()
    best_mol.AddConformer(mol.GetConformer(best_cid), assignId=True)
    Chem.AssignStereochemistryFrom3D(best_mol)
    if axis is not None:
        info["final_angle"] = _measure_dihedral(best_mol, axis)

    source = "smiles_no_axis_applied" if strategy == "plain_no_axis" else "smiles"
    return best_mol, source, info
