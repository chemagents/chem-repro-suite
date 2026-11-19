"""Add complete scientific summaries after verified fusion, leaving training/cache keys intact."""

import hashlib
import io

import pandas as pd

from ..common.artifacts import artifact_root, export_tables
from ..common.contracts import EvidenceNote
from ..common.datasets import implementation_sha256, table
from ..common.matched_validation import summarize_matched_predictions
from .data import load


def _artifact_bytes(result, sha, suffix):
    matches = [a for a in result.artifacts if a.sha256 == sha and a.name.endswith("." + suffix)]
    if not matches:
        raise ValueError("Complete result has no matching artifact")
    path = artifact_root() / (sha + "." + suffix)
    if path.is_symlink():
        raise ValueError("Artifact cannot be a symlink")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != sha or any(a.size_bytes != len(body) for a in matches):
        raise ValueError("Complete artifact checksum/size mismatch")
    return body


def attach_fusion_summary(result):
    metrics = result.answer.tables["paired_metrics"]
    if metrics.is_preview:
        records = pd.read_csv(io.BytesIO(_artifact_bytes(result, metrics.full_table_artifact_sha256, "csv")))
        if len(records) != metrics.total_rows:
            raise ValueError("Incomplete matched metric table")
        records = records.to_dict("records")
    else:
        records = metrics.rows
    predictions = {}
    suffix = "_fusion_heldout_predictions.csv"
    for artifact in result.artifacts:
        if artifact.name.endswith(suffix):
            scheme = artifact.name.removesuffix(suffix)
            if scheme in predictions:
                raise ValueError("Ambiguous held-out prediction artifact")
            frame = pd.read_csv(io.BytesIO(_artifact_bytes(result, artifact.sha256, "csv")))
            if not (abs(frame.product_score - frame.structure_probability * frame.rte_probability) <= 1e-12).all():
                raise ValueError("Stored product scores differ from their components")
            predictions[scheme] = frame
    source = load()
    labels = source.loc[source.activity.notna()].set_index("ligand_id").activity
    for frame in predictions.values():
        if not set(frame.ligand_id).issubset(labels.index):
            raise ValueError("Prediction IDs are outside the labeled source")
        if not (frame.activity.to_numpy() == labels.loc[frame.ligand_id].to_numpy()).all():
            raise ValueError("Prediction labels differ from the source")
    tables = summarize_matched_predictions(
        predictions, records,
        branches={"structure": "structure_probability", "rte": "rte_probability", "product": "product_score"},
        candidate="product", source_ids=source.loc[source.activity.notna(), "ligand_id"],
    )
    descriptions = {
        "fold_summary": "Unweighted mean and sample SD across completed folds/holdouts; confusion sums count test instances, not unique molecules. Not pooled OOF or confidence intervals.",
        "paired_summary": "Product minus each component on identical held-out records in every split; relative FPR reduction uses the ratio of mean FPRs.",
        "scoring_coverage": "Matched finite scores divided by declared test instances, and unique test IDs divided by the labeled source population. Neither is positive-selection retention or metabolome coverage.",
        "paired_fold_differences": "Complete paired differences for every split and both reference branches; no filtering by effect direction.",
    }
    for name, frame in tables.items():
        result.answer.tables[name] = table(frame, descriptions[name])
    result.metadata.protocol_parameters["matched_summary"] = {
        "version": 1, "metrics_recomputed_from_complete_predictions": True,
        "input_analysis_implementation_sha256": result.metadata.implementation_sha256,
        "aggregation": "unweighted split mean; sample SD; paired differences per split",
        "confusion_sum_unit": "held-out prediction instances; repeated molecules may occur",
    }
    result.metadata.implementation_sha256 = implementation_sha256()
    result.answer.finding += " Complete summaries compare the product with BOTH branches across every split; coverage uses explicit test-instance and source-record denominators."
    for scheme, frame in tables["fold_summary"].groupby("scheme", sort=True):
        result.answer.finding += " " + scheme + " mean FPR / recall: " + "; ".join(
            f"{row.branch} {row.mean_fpr:.6f} / {row.mean_recall:.6f}"
            for row in frame.itertuples()) + "."
    result.answer.limitations.append(EvidenceNote(
        code="PAPER_EVALUATION_THRESHOLD", severity="info",
        message="The article section4.7 explicitly reports0.5 for FPR performance evaluation; it is not a hard cutoff for metabolite ranking. "
                "The same0.5 is used in these reconstructed comparisons. This reported method does not establish identical author folds or models.",
    ))
    result.answer.limitations.append(EvidenceNote(
        code="FOLD_SUMMARY_SCOPE", severity="info",
        message="Repeated-holdout means and sample SD are descriptive. Shared molecules make split estimates dependent. "
                "Summed confusion counts are prediction instances, not counts of unique molecules; positive retention is separate from scoring coverage.",
    ))
    return export_tables(result, port=7341)
