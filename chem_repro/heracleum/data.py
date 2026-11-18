"""The master predictions and source identities are distinct, audited inputs."""

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from ..common.chemistry import canonical_smiles, scaffold
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, verified_manifest

ENDPOINTS = dict(
    zip(
        [
            "respiratory",
            "neurotoxicity",
            "nephrotoxicity",
            "ototoxicity",
            "hematotoxicity",
            "genotoxicity",
            "hepatotoxicity",
            "eye_corrosion",
            "eye_irritation",
            "skin_sensitization",
        ],
        [
            "admetlab_tox_" + n
            for n in [
                "respiratory",
                "neurotoxicity_di",
                "nephrotoxicity_di",
                "ototoxicity",
                "hematotoxicity",
                "genotoxicity",
                "h_ht",
                "ec",
                "ei",
                "skinsen",
            ]
        ],
    )
)
SATURATED = {"genotoxicity", "eye_irritation"}
CLASSES = ["Furo 1", "Furo 2", "Top 1", "Top 2", "Top 3"]


def source_dataset(subset):
    if subset not in {"DS1", "DS2"}:
        raise ValueError("subset must be DS1 or DS2")
    verified_manifest("heracleum")
    return pd.read_csv(DATA_ROOT / "heracleum" / ("DS1_2008.csv" if subset == "DS1" else "DS2_102.csv"))


@lru_cache(maxsize=1)
def _load():
    verified_manifest("heracleum")
    df = pd.read_csv(DATA_ROOT / "heracleum/DS1-DS2_df.csv")
    if df.synID.duplicated().any():
        raise ValueError("Master synID is not unique")
    names = source_dataset("DS2")[["synID", "trivial_name", "SMILES_standardized"]]
    refs = pd.concat(
        [source_dataset(s)[["synID", "SMILES_standardized"]] for s in ["DS1", "DS2"]]
    ).drop_duplicates()
    if refs.synID.duplicated().any():
        raise ValueError("Conflicting source structures for the same synID")
    df = df.merge(refs, on="synID", how="left", validate="one_to_one", sort=False)
    df = df.merge(names[["synID", "trivial_name"]], on="synID", how="left", validate="one_to_one", sort=False)
    df["identity_verified"] = df.canonical_smiles.map(canonical_smiles).eq(
        df.SMILES_standardized.map(canonical_smiles)
    )
    df["reported_name"] = df.trivial_name
    df.loc[~df.identity_verified, "trivial_name"] = None
    df["generic_scaffold_matches"] = df.canonical_smiles.map(
        lambda s: scaffold(s, generic=True, retain_stereo=True)
    ).eq(df.generic_murcko_scaffold)
    labels = [s for s, _ in Counter(df.murcko_scaffold).most_common(5)]
    df["scaffold_class"] = df.murcko_scaffold.map(dict(zip(labels, CLASSES))).fillna("Other")
    targets = [c for c in df if c.startswith("dock_") and c != "dock_pc1"]
    if len(targets) != 44:
        raise ValueError("The source must contain exactly 44 target score columns")
    tox = df[list(ENDPOINTS.values())].to_numpy(float)
    if not np.isfinite(tox).all() or ((tox < 0) | (tox > 1)).any():
        raise ValueError("Invalid probability table")
    for c in ["ds1_member", "ds2_member", "has_valid_44_docking_scores"]:
        if df[c].dtype != bool:
            raise ValueError(f"Invalid boolean source column {c}")
    valid = df.loc[df.has_valid_44_docking_scores, targets].to_numpy(float)
    if not np.isfinite(valid).all() or (valid == 0).all(axis=1).any():
        raise ValueError("Invalid valid-docking mask")
    return df, targets


def load(policy="published_snapshot"):
    df, targets = _load()
    if policy == "strict":
        df = df.loc[df.identity_verified]
    elif policy != "published_snapshot":
        raise ValueError("Unknown identity policy")
    return df.copy(), list(targets)


def limitations():
    df, _ = _load()
    n = int((~df.identity_verified).sum())
    notes = [
        EvidenceNote(
            code="MODEL_DERIVED_ENDPOINTS",
            message="ADMETlab values are predictions; docking "
            "scores are computed descriptors, not measured affinity. Their agreement is not independent "
            "biological validation.",
        )
    ]
    if n:
        notes.append(
            EvidenceNote(
                code="SOURCE_IDENTITY_CONFLICT",
                message=f"{n} master structures conflict "
                "with source dataset structures at the same synID. Published-snapshot results reproduce the deposited "
                "table despite these conflicts. Conflicting molecule names are withheld; no predictions are transplanted.",
            )
        )
    return notes
