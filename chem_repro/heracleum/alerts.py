"""Compare archived alerts with recomputation on both conflicting identity sources."""

from functools import lru_cache

import pandas as pd

from ..common.alerts import structural_alerts
from .data import load, source_dataset


@lru_cache(maxsize=1)
def alert_audit():
    master, _ = load()
    source = source_dataset("DS2").set_index("synID")
    if not source.index.is_unique:
        raise ValueError("Source DS2 IDs must be unique for an identity-specific alert audit")
    rows = []
    for row in master.loc[master.ds2_member].itertuples():
        original_smiles = source.loc[row.synID, "SMILES_standardized"]
        deposited = structural_alerts(row.canonical_smiles)
        original = structural_alerts(original_smiles)
        rows.append(
            {
                "synID": int(row.synID),
                "identity_verified": bool(row.identity_verified),
                "master_prediction_smiles": row.canonical_smiles,
                "original_DS2_smiles": original_smiles,
                "archived_pains_count": row.admetai_pains_alert,
                "archived_brenk_count": row.admetai_brenk_alert,
                **{"master_" + k: v for k, v in deposited.items()},
                **{"original_DS2_" + k: v for k, v in original.items()},
                "master_counts_match_archive": bool(
                    deposited["pains_count"] == row.admetai_pains_alert
                    and deposited["brenk_count"] == row.admetai_brenk_alert
                ),
                "original_DS2_counts_match_archive": bool(
                    original["pains_count"] == row.admetai_pains_alert
                    and original["brenk_count"] == row.admetai_brenk_alert
                ),
            }
        )
    return pd.DataFrame(rows)
