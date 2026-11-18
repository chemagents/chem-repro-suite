"""Reviewed Figure4 connectivity links, with no inferred stereochemistry or CCD annotation."""

import json
from hashlib import sha256

import numpy as np
import pandas as pd
from rdkit import Chem

from ..common.artifacts import save_artifact
from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .data import load


def stereo_free_key(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid candidate structure")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)  # retain isotopes and charges


def identity_map():
    verified_manifest("cannabis_candidates")
    verified_manifest("open_syntelly_2024")
    root = DATA_ROOT / "cannabis_candidates"
    mapping = json.loads((root / "identity_map.json").read_text())
    grid_hashes = [value for path, value in mapping["parents"].items() if path.endswith("/connectivity_review_grid.png")]
    if len(grid_hashes) != 1 or grid_hashes[0] != sha256((root / "identity_review_grid.png").read_bytes()).hexdigest():
        raise ValueError("Candidate depiction grid differs from its reviewed identity map")
    for name in ["cannabis/data.csv", "cannabis_candidates/paper.pdf",
                 "cannabis_candidates/figure4_candidates.csv", "open_syntelly_2024/cannabis_query.csv"]:
        recorded = [value for path, value in mapping["parents"].items() if path.endswith("chem_repro/data/" + name)]
        if len(recorded) != 1 or recorded[0] != sha256((DATA_ROOT / name).read_bytes()).hexdigest():
            raise ValueError("Candidate identity map belongs to different source records or paper")
    source = load().set_index("ligand_id")
    query = pd.read_csv(DATA_ROOT / "open_syntelly_2024/cannabis_query.csv", dtype={"query_id": str}).set_index("query_id")
    reference = pd.read_csv(root / "figure4_candidates.csv").set_index("paper_rank")
    rows = mapping["records"]
    if [row["paper_rank"] for row in rows] != list(range(1, 11)) or len({r["source_ligand_id"] for r in rows}) != 10:
        raise ValueError("Require all ten unique Figure4 source mappings in paper order")
    keys = source.SMILES.map(stereo_free_key)
    for row in rows:
        identifier = row["source_ligand_id"]
        if identifier not in source.index or source.loc[identifier, "SMILES"] != row["source_smiles"]:
            raise ValueError("Candidate source ID/SMILES changed; no name transfer allowed")
        if pd.notna(source.loc[identifier, "activity"]):
            raise ValueError("Figure4 query mapping unexpectedly points to a labeled reference")
        if row["toxicity_query_id"] not in query.index or query.loc[row["toxicity_query_id"], "smiles"] != row["source_smiles"]:
            raise ValueError("Toxicity-query ID/SMILES differs from the reviewed source mapping")
        key = stereo_free_key(row["source_smiles"])
        if key != row["connectivity_smiles"]:
            raise ValueError("Mapped connectivity differs from the original source molecule")
        if (reference.loc[row["paper_rank"], "paper_name"] != row["paper_name"]
                or reference.loc[row["paper_rank"], "paper_final_score"] != row["paper_final_score"]):
            raise ValueError("Caption identity or reported score differs from the reviewed figure")
        for cid in row["pubchem_cids"]:
            record = json.loads((root / f"pubchem_{cid}.json").read_text())
            if record["CID"] != cid or stereo_free_key(record["SMILES"]) != key:
                raise ValueError("PubChem snapshot does not support the mapped connectivity")
        labeled_matches = source.index[(keys == key) & source.activity.notna()].tolist()
        if sorted(labeled_matches) != sorted(row["labeled_reference_match_ids"]):
            raise ValueError("Recorded classification-reference overlap differs from current source identities")
    return rows


def identity_table(rows):
    return table(pd.DataFrame([{
        "paper_rank": row["paper_rank"], "paper_name": row["paper_name"],
        "source_ligand_id": row["source_ligand_id"], "source_smiles": row["source_smiles"],
        "pubchem_cids": row["pubchem_cids"], "labeled_reference_match_ids": row["labeled_reference_match_ids"],
        "identity_scope": "stereochemistry removed; isotopes/charge/tautomers retained",
    } for row in rows]), "Reviewed Figure4 depiction-to-source connectivity, not stereoisomer identity or biological occurrence", "source_record")


def activity_rows(rows, scores):
    if scores.ligand_id.isna().any() or scores.ligand_id.duplicated().any():
        raise ValueError("Require unique source IDs before naming candidate scores")
    indexed = scores.set_index("ligand_id")
    output = []
    for row in rows:
        identifier = row["source_ligand_id"]
        if identifier not in indexed.index or indexed.loc[identifier, "SMILES"] != row["source_smiles"]:
            raise ValueError("Candidate score is absent or belongs to a different source molecule")
        score = indexed.loc[identifier]
        probabilities = [float(score[name]) for name in ["structure_probability", "rte_probability", "product_score"]]
        if not np.isfinite(probabilities).all() or any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Invalid candidate component probabilities")
        if not np.isclose(probabilities[0] * probabilities[1], probabilities[2], rtol=0, atol=1e-12):
            raise ValueError("Candidate product is not the product of the declared mean component scores")
        ranks = np.asarray([score.structural_rank, score.product_rank_before_annotations,
                            score.rank_change_from_structure], dtype=float)
        if (not np.isfinite(ranks).all() or not np.equal(ranks, np.floor(ranks)).all()
                or not ((ranks[:2] >= 1) & (ranks[:2] <= len(scores))).all()
                or ranks[2] != ranks[0] - ranks[1]):
            raise ValueError("Candidate ranks are inconsistent with the complete source-cohort ranking")
        if not isinstance(score.structure_screen_positive, (bool, np.bool_)):
            raise TypeError("Structural screen result must be an explicit boolean")
        output.append({"paper_rank": row["paper_rank"], "paper_name": row["paper_name"], "source_ligand_id": identifier,
                       "computed_structure_probability": probabilities[0], "computed_rte_probability": probabilities[1],
                       "computed_product_score": probabilities[2],
                       "source_structural_rank": int(ranks[0]),
                       "source_product_rank_before_annotations": int(ranks[1]),
                       "source_rank_change_from_structure": int(ranks[2]),
                       "source_rank_change_direction": "improved" if ranks[2] > 0 else "worsened" if ranks[2] < 0 else "unchanged",
                       "source_rank_population_size": len(scores),
                       "source_rank_comparison": "structural_to_product_within_source_query_cohort; not_vs_paper_rank",
                       "source_structure_screen_positive": bool(score.structure_screen_positive),
                       "classification_training_connectivity_overlap": bool(row["labeled_reference_match_ids"])})
    return pd.DataFrame(output)


def toxicity_rows(rows, predictions, endpoint, spec):
    if predictions.query_id.isna().any() or predictions.query_id.duplicated().any():
        raise ValueError("Require unique toxicity-query IDs before naming predictions")
    indexed = predictions.set_index("query_id")
    result = []
    for row in rows:
        identifier = row["toxicity_query_id"]
        if identifier not in indexed.index or indexed.loc[identifier, "smiles"] != row["source_smiles"]:
            raise ValueError("Candidate toxicity prediction is absent or its query identity changed")
        score = indexed.loc[identifier]
        value, domain = float(score.prediction), float(score.reference_domain_score_percent)
        if not np.isfinite([value, domain]).all() or not 0 <= domain <= 100:
            raise ValueError("Invalid candidate prediction or relative domain score")
        if spec["task"] == "classification" and not 0 <= value <= 1:
            raise ValueError("Classification probability is outside [0,1]")
        flags = {name: score[name] for name in ["training_identity_overlap", "training_scaffold_overlap", "outside_reference_distance"]}
        if any(not isinstance(value, (bool, np.bool_)) for value in flags.values()):
            raise ValueError("Applicability/overlap flags must be explicit booleans, not strings or missing values")
        result.append({"paper_rank": row["paper_rank"], "paper_name": row["paper_name"],
                       "source_ligand_id": row["source_ligand_id"], "source_smiles": row["source_smiles"],
                       "query_id": identifier, "endpoint": endpoint, "task": spec["task"], "unit": spec["unit"],
                       "prediction": value, "domain_score_percent": domain,
                       **{name: bool(value) for name, value in flags.items()}})
    return result


def toxicity_tables(records):
    frame = pd.DataFrame(records)
    keys = ["paper_rank", "paper_name", "source_ligand_id"]
    if frame.duplicated(["paper_rank", "endpoint"]).any() or not frame.groupby("endpoint").size().eq(10).all():
        raise ValueError("Every requested endpoint must cover all ten named source molecules exactly once")
    tables = {}
    for column, name in [("prediction", "figure4_prediction_matrix"), ("domain_score_percent", "figure4_domain_score_matrix"),
                         ("training_identity_overlap", "figure4_training_identity_overlap")]:
        pivot = frame.pivot(index=keys, columns="endpoint", values=column).reset_index()
        tables[name] = table(pivot, "New source-model values for mapped Figure4 connectivity; domain percentage is not calibrated accuracy")
    tables["figure4_endpoint_units"] = table(frame[["endpoint", "task", "unit"]].drop_duplicates(), "Endpoint-specific source units; no conversion of unverified aquatic scales")
    return frame, tables


def attach_mapping_provenance(output, *, base_url):
    sources = provenance("cannabis_candidates", "figure4.reviewed_connectivity.v1").sources
    present = {(s.name, s.sha256) for s in output.metadata.sources}
    output.metadata.sources.extend(s for s in sources if (s.name, s.sha256) not in present)
    output.metadata.protocol_parameters["figure4_mapping"] = {
        "identity_map_sha256": verified_manifest("cannabis_candidates")["files"]["identity_map.json"]["sha256"],
        "stereochemistry_inferred": False, "CCD_annotation_inferred": False, "author_component_scores_recovered": False,
    }
    path = DATA_ROOT / "cannabis_candidates/identity_map.json"
    output.artifacts.append(save_artifact(path.read_bytes(), suffix="json", media_type="application/json",
                                          public_base_url=base_url, display_name="figure4_identity_map_and_provenance.json"))
    output.artifacts.append(save_artifact((DATA_ROOT / "cannabis_candidates/identity_review_grid.png").read_bytes(),
                                          suffix="png", media_type="image/png", public_base_url=base_url,
                                          display_name="figure4_reviewed_source_structures.png"))
    output.answer.limitations.append(EvidenceNote(code="FIGURE4_CONNECTIVITY_SCOPE", message=
        "The ten displayed Figure4 graphs were manually reviewed and matched to deposited source rows and PubChem "
        "connectivity snapshots. Stereochemistry, the full 44-compound CCD-filtered set and independent natural occurrence are "
        "not established by this mapping. New source-model component/toxicity scores are not original author outputs. "
        "Rank 1 shares connectivity with pesticide reference 1154 and rank 9 with AChE-inactive reference 3230 in this "
        "source bundle; neither is independent validation of these source classifiers."))
    return output
