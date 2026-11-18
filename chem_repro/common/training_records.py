"""Auditable identity-level aggregation; no salt, tautomer or stereo replacement."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass
class PreparedTraining:
    molecules: pd.DataFrame
    records: pd.DataFrame
    summary: dict


def prepare_training_records(frame, *, task, allowed_elements, regression_aggregation="record_mean"):
    """Preserve every raw record and expose both regression weighting estimands.

    source_dataset denotes a data source, never a per-record source_id. The caller
    must declare whether these are original experimental sources or only collection
    labels. Canonical isomeric identity retains isotopes, charges and all fragments;
    atom-map numbers are removed because they are annotations, not molecular identity.
    """
    required = {"source_id", "source_dataset", "smiles", "y"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("Require nonempty records with source_id, source_dataset, smiles and y")
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown target task")
    if regression_aggregation not in {"record_mean", "source_balanced_mean"}:
        raise ValueError("Choose a declared regression weighting estimand")
    if frame.source_dataset.isna().any():
        raise ValueError("Source dataset cannot be missing")
    if (frame.source_dataset.astype(str).str.strip() == "").any():
        raise ValueError("Source dataset cannot be blank")
    y = pd.to_numeric(frame.y, errors="raise").to_numpy(float)
    if not np.isfinite(y).all() or (task == "classification" and not set(y) <= {0.0, 1.0}):
        raise ValueError("Targets must be finite, with binary 0/1 classification labels")
    allowed = frozenset(allowed_elements)
    if "C" not in allowed:
        raise ValueError("The organic training policy must allow carbon")
    records = []
    for number, (_, row) in enumerate(frame.iterrows()):
        raw = row.smiles
        mol = Chem.MolFromSmiles(raw) if isinstance(raw, str) else None
        reasons, canonical, maps_removed = [], None, False
        if mol is None or not mol.GetNumAtoms():
            reasons.append("invalid_or_empty_smiles")
        else:
            elements = {a.GetSymbol() for a in mol.GetAtoms()}
            if "C" not in elements:
                reasons.append("no_carbon")
            unsupported = sorted(elements - allowed - {"H"})
            if unsupported:
                reasons.append("unsupported_elements:" + ",".join(unsupported))
            for atom in mol.GetAtoms():
                maps_removed |= bool(atom.GetAtomMapNum())
                atom.SetAtomMapNum(0)
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        missing_id = pd.isna(row.source_id) or str(row.source_id).strip() == ""
        records.append({"input_row": number, "source_id": None if missing_id else str(row.source_id),
                        "source_id_missing": bool(missing_id),
                        "source_dataset": str(row.source_dataset), "raw_smiles": raw,
                        "raw_y": float(y[number]), "canonical_smiles": canonical,
                        "atom_maps_removed": maps_removed,
                        "status": "chemistry_rejected" if reasons else "eligible",
                        "reason": ";".join(reasons), "molecule_id": None})
    ledger = pd.DataFrame(records)
    molecules = []
    for identity, group in ledger.loc[ledger.status == "eligible"].groupby("canonical_smiles", sort=True):
        if task == "classification" and group.raw_y.nunique() > 1:
            ledger.loc[group.index, ["status", "reason"]] = ["label_conflict", "conflicting_binary_labels"]
            continue
        identifier = hashlib.sha256(identity.encode()).hexdigest()
        record_mean = float(group.raw_y.mean())
        source_mean = float(group.groupby("source_dataset").raw_y.mean().mean())
        value = record_mean if regression_aggregation == "record_mean" else source_mean
        ledger.loc[group.index, "status"] = "retained_in_aggregate"
        ledger.loc[group.index, "molecule_id"] = identifier
        molecules.append({"molecule_id": identifier, "smiles": identity,
                          "y": value, "record_mean": record_mean, "source_balanced_mean": source_mean,
                          "n_records": len(group), "n_sources": int(group.source_dataset.nunique()),
                          "input_rows_json": json.dumps(group.input_row.tolist()),
                          "source_ids_json": json.dumps(group.source_id.tolist()),
                          "source_datasets_json": json.dumps(sorted(group.source_dataset.unique()))})
    columns = ["molecule_id", "smiles", "y", "record_mean", "source_balanced_mean", "n_records",
               "n_sources", "input_rows_json", "source_ids_json", "source_datasets_json"]
    result = pd.DataFrame(molecules, columns=columns)
    retained = int((ledger.status == "retained_in_aggregate").sum())
    summary = {"input_records": len(ledger), "chemistry_rejected_records": int((ledger.status == "chemistry_rejected").sum()),
               "missing_source_id_records": int(ledger.source_id_missing.sum()),
               "conflicting_label_records": int((ledger.status == "label_conflict").sum()),
               "retained_input_records": retained, "aggregated_molecules": len(result),
               "duplicate_records_collapsed": retained - len(result),
               "regression_aggregation": regression_aggregation if task == "regression" else None,
               "original_records_preserved_in_ledger": True}
    return PreparedTraining(result, ledger, summary)
