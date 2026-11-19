"""Pinned source inputs, identities and split alignment for Cannabis."""

from functools import lru_cache

import pandas as pd

from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, verified_manifest

TARGETS = {
    "GSTE2": ("2imi", "7165_ANOGA_gste2_PDB-2IMI_residue_matrix.csv"),
    "ACHE": ("d8v7j0", "32264_TETUR_ace_UP-D8V7J0_residue_matrix.csv"),
    "GLC1": ("3rif", "6239_CAEEL_glc1_PDB-3RIF_residue_matrix.csv"),
    "UGT202A2": ("8sfy", "32264_TETUR_ugt202a2_PDB-8SFY_residue_matrix.csv"),
    "GSTM12": ("8udb", "32264_TETUR_gstm12_PDB-8UDB_residue_matrix.csv"),
    "OR28": ("8v3d", "7165_ANOGA_agamor28_PDB-8V3D_residue_matrix.csv"),
}


def limitations():
    return [
        EvidenceNote(
            code="SOURCE_COHORT_MISMATCH",
            message="Deposited data.csv contains 2749 unlabeled "
            "records; final-paper docked DS1 contains 2465, and standardized DS1 contains 5027. "
            "A local victor.csv now supplies2465 exact source structures and1010 archived scores >=0.70 "
            "(see recovered_author_cohort), but its producing model and final-paper lineage remain unverified. "
            "Unlabeled-source summaries still use2749 rows unless explicitly labeled otherwise.",
        ),
        EvidenceNote(
            code="OPERATIONAL_NEGATIVE_CLASS",
            message="Label 0 means the DS3 AChE-inactive "
            "reference class. It does not establish inactivity against other pesticidal targets, nor "
            "absence of pesticidal activity. Model scores are not experimental biopesticide efficacy.",
        ),
    ]


@lru_cache(maxsize=1)
def load():
    verified_manifest("cannabis")
    df = pd.read_csv(DATA_ROOT / "cannabis/data.csv")
    if df.ligand_id.isna().any() or df.ligand_id.duplicated().any():
        raise ValueError("Ligand IDs must be unique and nonmissing")
    if not set(df.activity.dropna().unique()).issubset({0, 1}):
        raise ValueError("Unexpected activity labels")
    return df


def split(split_index):
    if not 0 <= split_index < 30:
        raise ValueError("Source split index must be 0..29")
    df = load().set_index("ligand_id", drop=False)
    registry = pd.read_csv(DATA_ROOT / "cannabis/split_registry.csv")
    r = registry.loc[registry.split_index == split_index].copy()
    if r.empty or r.ligand_id.duplicated().any() or set(r["set"]) != {"train", "test"}:
        raise ValueError("Source split is missing, duplicated or malformed")
    if set(r.ligand_id) != set(df.loc[df.activity.notna()].index):
        raise ValueError("Split must cover the entire labeled cohort exactly once")
    if (df.loc[r.ligand_id, "activity"].to_numpy() != r.activity.to_numpy()).any():
        raise ValueError("Split labels differ from source data")
    train = df.loc[r.loc[r["set"] == "train", "ligand_id"]].copy()
    test = df.loc[r.loc[r["set"] == "test", "ligand_id"]].copy()
    return train, test


@lru_cache(maxsize=6)
def residue_matrix(target):
    if target not in TARGETS:
        raise ValueError("Unknown target")
    verified_manifest("cannabis")
    frame = pd.read_csv(DATA_ROOT / "cannabis" / TARGETS[target][1])
    if frame.ligand_id.isna().any() or frame.ligand_id.duplicated().any():
        raise ValueError("Residue matrix IDs must be unique and nonmissing")
    return frame.set_index("ligand_id")
