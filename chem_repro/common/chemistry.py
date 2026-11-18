"""Explicit chemical identities and representations; no implicit salt/stereo stripping."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


@lru_cache(maxsize=32768)
def canonical_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Unparseable or empty molecular structure")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def catalyst_identity(smiles: str, axial_configuration: str = "none") -> tuple[str, str]:
    axial = str(axial_configuration).strip()
    if axial not in {"R", "S", "none"}:
        raise ValueError("Axial configuration must be R, S or none; missing is not none")
    return canonical_smiles(smiles), axial


def scaffold(smiles: str, *, generic: bool = False, retain_stereo: bool = False) -> str:
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        core = MurckoScaffold.MakeScaffoldGeneric(core)
    if not retain_stereo:
        Chem.RemoveStereochemistry(core)
    # Empty string is an acyclic structure, never replaced by the full molecule silently.
    return Chem.MolToSmiles(core, isomericSmiles=retain_stereo)


def morgan_fingerprints(smiles: list[str], *, chirality: bool, radius: int = 2, nbits: int = 2048):
    if radius < 0 or nbits < 8:
        raise ValueError("Invalid fingerprint dimensions")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits, includeChirality=chirality)
    return [gen.GetFingerprint(Chem.MolFromSmiles(canonical_smiles(s))) for s in smiles]


def tanimoto_distances(fingerprints) -> np.ndarray:
    """RDKit bit operations avoid uint8 dot-product overflow on large structures."""
    n = len(fingerprints)
    dist = np.zeros((n, n), dtype=float)
    for i, fp in enumerate(fingerprints):
        dist[i] = 1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fingerprints))
    np.fill_diagonal(dist, 0.0)
    return dist


def ee_to_ddg(
    ee_percent: float,
    temperature_celsius: float,
    *,
    ee_cap_percent: float = 99.9,
    gas_constant_kcal: float = 0.001987,
) -> float:
    """Magnitude |ΔΔG‡|, kcal/mol. ee and dr are not interchangeable.

    The source dataset uses the rounded R=0.001987 kcal mol−1 K−1. R is explicit
    so an SI-precision alternative is a different, recorded calculation.
    """
    vals = [ee_percent, temperature_celsius, ee_cap_percent, gas_constant_kcal]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("ee, temperature and constants must be finite")
    if not 0 <= ee_percent <= 100 or not 0 < ee_cap_percent < 100:
        raise ValueError("Magnitude ee must be in [0,100] and cap strictly between 0 and 100")
    kelvin = temperature_celsius + 273.15
    if kelvin <= 0 or gas_constant_kcal <= 0:
        raise ValueError("Temperature must exceed absolute zero; R must be positive")
    fraction = min(ee_percent, ee_cap_percent) / 100
    return gas_constant_kcal * kelvin * (math.log1p(fraction) - math.log1p(-fraction))
