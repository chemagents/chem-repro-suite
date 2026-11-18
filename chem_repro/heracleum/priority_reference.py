"""Compare calculated priorities with the pinned paper without relabelling structures."""

import json
import os

import pandas as pd
from rdkit import Chem

from ..common.artifacts import save_artifact
from ..common.chemistry import canonical_smiles
from ..common.comparison import compare_identifier_sets
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .data import load


def _key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(canonical_smiles(smiles)))


def _label(value):
    return str(value) if pd.notna(value) and str(value).strip() else None


def attach_priority_reference(result):
    """Append comparisons to complete candidate rows before bounded table export."""
    if "tier1_reference_agreement" in result.answer.tables:
        return result
    candidates = result.answer.tables["candidates"]
    if candidates.is_preview:
        raise ValueError("Priority-reference comparison requires all candidate rows")
    manifest = verified_manifest("heracleum")
    paper_manifest = verified_manifest("heracleum_paper")
    reference = pd.read_csv(DATA_ROOT / "heracleum/table_s4_tier1_keys.csv", keep_default_na=False)
    claim = json.loads((DATA_ROOT / "heracleum_paper/priority_reference.json").read_text())
    if claim["parent_sha256"] != paper_manifest["files"]["paper.pdf"]["sha256"]:
        raise ValueError("Priority claim belongs to another paper")
    if claim["identity_source_sha256"] != manifest["files"]["supplement.pdf"]["sha256"]:
        raise ValueError("Priority claim identity belongs to another supplement")
    if reference.inchikey.duplicated().any() or reference.table_row.tolist() != list(range(1, len(reference) + 1)):
        raise ValueError("Ambiguous published priority reference")
    master, _ = load("published_snapshot")
    source = master.loc[master.ds2_member].copy()
    source["source_key"] = source.SMILES_standardized.map(_key)
    source["master_key"] = source.canonical_smiles.map(_key)
    if source.source_key.duplicated().any():
        raise ValueError("Published keys do not resolve to unique DS2 records")
    by_key = {r.source_key: r for r in source.itertuples()}
    by_id = {int(r.synID): r for r in source.itertuples()}
    calculated = {row["synID"]: row for row in candidates.rows}
    if len(calculated) != len(candidates.rows) or set(calculated) != set(by_id):
        raise ValueError("Candidate rows do not cover DS2 exactly once")
    selected = [by_id[row["synID"]] for row in candidates.rows if row["tier"] == 1]
    scopes = [
        ("record_associated_DS2_structures", {r.source_key for r in selected},
         "Full InChIKey of original DS2 structure attached to each selected synID; does not verify prediction identity"),
        ("prediction_master_structures", {r.master_key for r in selected},
         "Full InChIKey of the master prediction structure at each selected synID"),
    ]
    agreements = [{"scope": scope, "selected_record_count": len(selected),
                   **compare_identifier_sets(reference.inchikey, keys, identity_definition=definition)}
                  for scope, keys, definition in scopes]
    mappings = []
    for row in reference.itertuples():
        original = by_key.get(row.inchikey)
        computed = calculated[int(original.synID)] if original is not None else None
        mappings.append({
            "table_s4_row": int(row.table_row), "pdf_page": int(row.pdf_page),
            "reference_inchikey": row.inchikey, "table_s4_name": row.name_as_printed or None,
            "synID": int(original.synID) if original is not None else None,
            "ds2_source_name": _label(original.reported_name) if original is not None else None,
            "master_inchikey": original.master_key if original is not None else None,
            "identity_verified": bool(original.identity_verified) if original is not None else None,
            "computed_tier": computed["tier"] if computed is not None else None,
            "n_favorable": computed["n_favorable"] if computed is not None else None,
        })
    target = by_key.get(claim["compound_inchikey"])
    observed = calculated[int(target.synID)] if target is not None else None
    medians = {r["endpoint"]: r["median"] for r in result.answer.tables["median_thresholds"].rows}
    all_eight = [row for row in candidates.rows if row["n_favorable"] == claim["n_favorable"]]
    matches = (observed is not None and observed["n_favorable"] == claim["n_favorable"]
               and (not claim["unique_in_sample"] or len(all_eight) == 1))
    result.answer.tables["tier1_reference_agreement"] = table(pd.DataFrame(agreements),
        "Two explicitly different identity comparisons against Supplementary Table S4; no acceptance threshold inferred")
    result.answer.tables["tier1_reference_membership"] = table(pd.DataFrame(mappings),
        "Published names and original DS2 names kept separately; original names are not assigned to conflicting master predictions")
    result.answer.tables["paper_reported_priority_claim"] = table(pd.DataFrame([{
        k: claim[k] for k in ["compound_name", "compound_inchikey", "n_favorable", "unique_in_sample", "pdf_pages", "source_sections"]
    }]), "Main-paper priority statement; structure identified through Table S4, not inferred from a name", evidence_kind="paper_reported")
    result.answer.tables["priority_claim_check"] = table(pd.DataFrame([{
        "reference_compound": claim["compound_name"],
        "synID": int(target.synID) if target is not None else None,
        "identity_verified": bool(target.identity_verified) if target is not None else None,
        "observed_n_favorable": observed["n_favorable"] if observed is not None else None,
        "matches_published_claim": matches,
        "probabilities": observed["probabilities"] if observed is not None else None,
        "DS2_median_thresholds": medians,
        "unfavorable_endpoints": [e for e, threshold in medians.items()
                                   if observed["probabilities"][e] >= threshold] if observed is not None else None,
    }]), "Recomputed count under the declared all-102 DS2 median rule, including the endpoint responsible for disagreement")
    result.answer.tables["all_informative_endpoints_favorable"] = table(pd.DataFrame([{
        "synID": row["synID"], "ds2_source_name": _label(by_id[row["synID"]].reported_name),
        "table_s4_names": reference.loc[reference.inchikey == by_id[row["synID"]].source_key, "name_as_printed"].tolist(),
        "source_inchikey": by_id[row["synID"]].source_key, "identity_verified": row["identity_verified"],
        "n_favorable": row["n_favorable"], "tier": row["tier"],
    } for row in all_eight], columns=["synID", "ds2_source_name", "table_s4_names", "source_inchikey", "identity_verified", "n_favorable", "tier"]),
        "Records favorable on all informative endpoints; distinct source labels preserved without resolving nomenclature")
    result.answer.finding += (
        f" Tier-1 agreement with Table S4: Jaccard {agreements[0]['jaccard']:.3f} by record-associated DS2 "
        f"structures and {agreements[1]['jaccard']:.3f} by master prediction structures."
    )
    result.answer.limitations.append(EvidenceNote(code="PRIORITY_REFERENCE_IDENTITY_SCOPE",
        message="Agreement by record-associated DS2 structures does not resolve master/source identity conflicts. "
                "The separate master-structure comparison tests the structures actually linked to predictions. "
                "Names from Table S4 and DS2 remain source labels; differing labels are not automatically resolved as synonyms."))
    if not matches:
        actual = observed["n_favorable"] if observed is not None else "unavailable"
        result.answer.finding += f" Published all-eight claim for {claim['compound_name']} disagrees with the archived-data count ({actual})."
        result.answer.limitations.append(EvidenceNote(code="PUBLISHED_PRIORITY_CLAIM_DISAGREEMENT",
            message=f"The paper names {claim['compound_name']} as uniquely favorable on all {claim['n_favorable']} endpoints; "
                    f"the source-data calculation gives {actual}. Inspect priority_claim_check and all_informative_endpoints_favorable. "
                    "Do not change identities, labels or thresholds to force agreement, or treat the paper statement as a computed result."))
    result.metadata.sources.extend(s for s in provenance("heracleum_paper", "heracleum.priority_reference.v1").sources
                                   if s.name in {"paper.pdf", "priority_reference.json"})
    result.metadata.protocol_parameters["priority_reference"] = {
        "supplement_table": "S4", "supplement_pdf_pages": [12, 13],
        "supplement_sha256": manifest["files"]["supplement.pdf"]["sha256"],
        "paper_sha256": claim["parent_sha256"], "paper_pdf_pages": claim["pdf_pages"],
        "comparison_identity": "full InChIKey, two separate structure sources", "thresholds_changed": False,
    }
    for study, filename in [("heracleum", "supplement.pdf"), ("heracleum_paper", "paper.pdf")]:
        result.artifacts.append(save_artifact((DATA_ROOT / study / filename).read_bytes(), suffix="pdf",
            media_type="application/pdf", public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7342"),
            display_name="heracleum_priority_reference_" + filename))
    return result
