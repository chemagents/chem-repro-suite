"""Audit the deposited table without claiming to repeat manual reaction-image curation."""

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..common.datasets import table
from ..common.record_audit import heavy_atom_balance, parse_reported_ratio, structure_diagnostics

ROLES = ["electrophile", "nucleophile", "organocatalyst", "product"]
NUMERIC_RULES = {
    "ee": (0, 100, False),
    "yield": (0, 100, False),
    "ddG_kcal_mol": (0, None, False),
    "temperature": (-273.15, None, True),
    "time": (0, None, False),
    "organocatalyst_concentration(mol%)": (0, None, False),
    "acid_additive_concentration(mol%)": (0, None, False),
    "base_additive_concentration(mol%)": (0, None, False),
    "additive_concentration(mol%)": (0, None, False),
    "water_additive_equiv": (0, None, False),
    "electrophile/nucleophile": (0, None, True),
}


def audit_records(df):
    generated = {"reaction_id", "normalized_doi", "catalyst_identity", *(r + "_canonical" for r in ROLES)}
    source_columns = [c for c in df if c not in generated]
    raw = json.loads(df[source_columns].to_json(orient="records"))
    conditions = [
        c
        for c in source_columns
        if c
        not in {
            "paper_doi",
            *ROLES,
            "ee",
            "ddG_kcal_mol",
            "dr",
            "syn/anti",
            "yield",
            "organocatalyst_3d_sdf",
            "organocatalyst_class",
        }
    ]
    # Record fields are compared, not inferred experimental replicates or author identities.
    signatures = defaultdict(list)
    exact = df[source_columns].duplicated(keep=False).to_numpy()
    records = []
    counts = Counter()
    range_rows = []
    for field, (lower, upper, strict) in NUMERIC_RULES.items():
        x = pd.to_numeric(df[field], errors="coerce")
        present = df[field].notna()
        invalid = present & (~np.isfinite(x) | (x <= lower if strict else x < lower))
        if upper is not None:
            invalid |= present & (x > upper)
        finite = x[np.isfinite(x)]
        range_rows.append(
            {
                "field": field,
                "n_present": int(present.sum()),
                "n_missing": int((~present).sum()),
                "n_invalid": int(invalid.sum()),
                "lower": lower,
                "lower_exclusive": strict,
                "upper": upper,
                "min_observed": float(finite.min()) if len(finite) else None,
                "max_observed": float(finite.max()) if len(finite) else None,
            }
        )
    for position, (source, (_, row)) in enumerate(zip(raw, df.iterrows()), 1):
        flags = []
        role_info = {role: structure_diagnostics(source[role]) for role in ROLES}
        product_info = role_info["product"]
        if product_info["unassigned_carbon_stereocenters"]:
            flags.append("product_unspecified_carbon_stereo")
        for role, info in role_info.items():
            if info["fragments"] > 1:
                flags.append(role + "_multiple_fragments")
        for field, (lower, upper, strict) in NUMERIC_RULES.items():
            value = source[field]
            if value is None:
                continue
            invalid = not isinstance(value, (float, int)) or not np.isfinite(value)
            if not invalid:
                invalid = value <= lower if strict else value < lower
                invalid |= upper is not None and value > upper
            if invalid:
                flags.append("invalid_numeric:" + field)
        for field in ["two_or_three_components", "molecular_sieves"]:
            if source[field] not in [0, 1]:
                flags.append("invalid_boolean:" + field)
        if source["syn/anti"] not in [None, "syn", "anti"]:
            flags.append("invalid_relative_configuration")
        ratio = None
        if source["dr"] is not None:
            try:
                ratio = parse_reported_ratio(source["dr"])
            except ValueError:
                flags.append("invalid_dr_format")
        for additive in ["acid_additive", "base_additive", "additive"]:
            named = source[additive] is not None
            loaded = source[additive + "_concentration(mol%)"] is not None
            if named and not loaded:
                flags.append(additive + "_loading_unreported")
            if loaded and not named:
                flags.append(additive + "_identity_unreported")
        balance = heavy_atom_balance([source["electrophile"], source["nucleophile"]], source["product"])
        if balance["heavy_atom_delta"]:
            flags.append("heavy_atom_inventory_difference")
        if balance["formal_charge_delta"]:
            flags.append("formal_charge_inventory_difference")
        if exact[position - 1]:
            flags.append("exact_duplicate_source_record")
        signature = {c: source[c] for c in conditions}
        signature.update(
            paper_doi=row.normalized_doi,
            electrophile=row.electrophile_canonical,
            nucleophile=row.nucleophile_canonical,
            organocatalyst=row.organocatalyst_canonical,
        )
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()
        signatures[digest].append(position - 1)
        records.append(
            {
                "reaction_id": row.reaction_id,
                "source_row_1based": position,
                "csv_line": position + 1,
                "source_publication": row.normalized_doi,
                "input_condition_signature": digest,
                "variant": "three_component" if source["two_or_three_components"] else "two_component",
                "product_unspecified_carbon_atom_indices": product_info["unassigned_carbon_stereocenters"],
                "role_fragment_counts": {role: r["fragments"] for role, r in role_info.items()},
                "role_formal_charges": {role: r["formal_charge"] for role, r in role_info.items()},
                "dr_parsed": ratio,
                **balance,
                "flag_codes": flags,
            }
        )
    differing_groups = []
    for digest, indexes in signatures.items():
        if len(indexes) < 2:
            continue
        outcome_fields = ["product", "ee", "yield", "dr", "syn/anti"]
        outcomes = {json.dumps({k: raw[i][k] for k in outcome_fields}, sort_keys=True) for i in indexes}
        if len(outcomes) > 1:
            differing_groups.append(
                {
                    "input_condition_signature": digest,
                    "n_records": len(indexes),
                    "reaction_ids": [records[i]["reaction_id"] for i in indexes],
                    "n_distinct_recorded_outcomes": len(outcomes),
                }
            )
            for i in indexes:
                records[i]["flag_codes"].append("same_recorded_inputs_different_outcomes")
    for record in records:
        counts.update(record["flag_codes"])
    full = pd.DataFrame(records)
    summaries = pd.DataFrame(
        [{"flag": code, "n_records": n} for code, n in sorted(counts.items())], columns=["flag", "n_records"]
    )
    tables = {
        "source_field_missingness": table(
            pd.DataFrame(
                [
                    {
                        "field": field,
                        "n_missing": int(df[field].isna().sum()),
                        "n_present": int(df[field].notna().sum()),
                    }
                    for field in source_columns
                ]
            ),
            "Missing means unrecorded; absence of an optional additive is not imputed",
        ),
        "numeric_field_audit": table(
            pd.DataFrame(range_rows), "Physical/value-domain checks; mol% above100 is allowed"
        ),
        "quality_flag_counts": table(
            summaries, "Diagnostics, not automatic exclusions or declarations of invalid chemistry"
        ),
        "repeated_input_groups": table(
            pd.DataFrame(
                differing_groups,
                columns=[
                    "input_condition_signature",
                    "n_records",
                    "reaction_ids",
                    "n_distinct_recorded_outcomes",
                ],
            ),
            "Same recorded inputs and conditions within a publication; missing experimental details can explain differing outcomes",
        ),
        "quality_flags_preview": table(
            full.loc[full.flag_codes.map(bool)].head(30),
            "First30 flagged records; full4015-row audit is exported separately",
        ),
    }
    return tables, full
