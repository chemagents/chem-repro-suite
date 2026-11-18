"""Keep reconstructed source ranking distinct from the paper's CCD-filtered final cohort."""


import os

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from ..common.artifacts import export_tables
from ..common.chemistry import canonical_smiles
from ..common.contracts import EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from .analysis import metric, result
from .candidate_identity import activity_rows, attach_mapping_provenance, identity_map, identity_table
from .data import load
from .fusion import fusion_runs


class CandidateAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ligand_id: int
    source_smiles: str = Field(min_length=1)
    is_secondary_metabolite: StrictBool
    annotation_source: str = Field(min_length=1)
    name: str | None = None
    occurrence_evidence: str | None = None


def annotate_and_rank(scores, annotations, *, cutoff=.70):
    if scores.ligand_id.duplicated().any():
        raise ValueError("Candidate source IDs must be unique")
    frame = scores.copy()
    frame["structure_screen_positive"] = frame.structure_probability >= cutoff
    frame["structural_rank"] = frame.structure_probability.rank(method="min", ascending=False).astype(int)
    frame["product_rank_before_annotations"] = frame.product_score.rank(method="min", ascending=False).astype(int)
    frame["rank_change_from_structure"] = frame.structural_rank - frame.product_rank_before_annotations
    frame["rank_change_direction"] = frame.rank_change_from_structure.map(
        lambda delta: "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged")
    frame["rank_population_size"] = len(frame)
    records = [CandidateAnnotation.model_validate(value).model_dump() for value in annotations]
    if len({r["ligand_id"] for r in records}) != len(records):
        raise ValueError("Duplicate candidate annotations")
    lookup = frame.set_index("ligand_id")
    for record in records:
        if record["ligand_id"] not in lookup.index:
            raise ValueError("Annotation refers to a molecule outside the available query cohort")
        if canonical_smiles(record["source_smiles"]) != canonical_smiles(lookup.loc[record["ligand_id"], "SMILES"]):
            raise ValueError("Annotation SMILES disagrees with the source molecule; no name transfer allowed")
    columns = ["ligand_id", "is_secondary_metabolite", "annotation_source", "name", "occurrence_evidence"]
    annotations_frame = pd.DataFrame(records, columns=columns)
    frame = frame.merge(annotations_frame, on="ligand_id", how="left", validate="one_to_one")
    frame["annotation_status"] = frame.is_secondary_metabolite.map({True:"secondary", False:"not_secondary"}).fillna("unknown")
    frame["eligible"] = frame.structure_screen_positive & frame.is_secondary_metabolite.eq(True)
    frame["final_rank_with_supplied_annotations"] = None
    selected = frame.loc[frame.eligible].sort_values(["product_score", "ligand_id"], ascending=[False, True])
    for rank, index in enumerate(selected.index, 1):
        frame.loc[index, "final_rank_with_supplied_annotations"] = rank
    return frame.sort_values(["product_score", "ligand_id"], ascending=[False, True]).reset_index(drop=True)


def summarize_mapped_references(components, *, source_population, scheme):
    """Summarize the mapped reference subset, never relabel it as a selected top ten."""
    if components.empty or not components.source_rank_population_size.eq(source_population).all():
        raise ValueError("Mapped reference ranks require one declared source population")
    n = len(components)
    positive = int(components.source_structure_screen_positive.sum())
    low = int(components.source_product_rank_before_annotations.min())
    high = int(components.source_product_rank_before_annotations.max())
    best = float(components.computed_product_score.max())
    names = components.loc[components.computed_product_score.eq(best), "paper_name"].tolist()
    population = f"{n} mapped Figure4 reference compounds under the {scheme} reconstructed ensemble; not a selected source top10"
    values = {"mapped_reference_records": n,
              "mapped_reference_screen_positive": positive,
              "mapped_reference_screen_negative": n - positive,
              "mapped_reference_product_rank_min": low,
              "mapped_reference_product_rank_max": high,
              "mapped_reference_product_score_min": float(components.computed_product_score.min()),
              "mapped_reference_product_score_max": best}
    metrics = [Metric(name=key, value=value, population=population) for key, value in values.items()]
    finding = (f"Among the {n} mapped Figure4 reference compounds, {positive} pass and {n-positive} fail the paper's "
               f"0.70 structural screen under the reconstructed {scheme} ensemble. Their product ranks span "
               f"{low}–{high} among all {source_population} source query records, before annotations. "
               f"The highest product within this mapped subset is {best:.6f} ({', '.join(names)}).")
    return metrics, finding


def prioritize_candidates(annotations=None, *, reference_only=False, scheme="random", max_epochs=60, iterations=500):
    verified_manifest("cannabis_candidates")
    reference = pd.read_csv(DATA_ROOT / "cannabis_candidates/figure4_candidates.csv")
    mapped = identity_map()
    tables = {"paper_top10": table(reference, "Figure4 caption names and rendered score labels; not recomputed outputs",
                                  evidence_kind="paper_reported"), "figure4_source_identity": identity_table(mapped)}
    notes = [EvidenceNote(code="PUBLISHED_RANKING_BOUNDARY", message="Figure4 gives the ten names and rounded final scores. "
                          "It does not give separate original DMPNN/RTE probabilities or the44eligible CCD annotation records. "
                          "The ten depicted graphs are now linked to source rows by reviewed connectivity; this does not recover "
                          "the complete final cohort. The published rounded products are known; their original two "
                          "factors and exact unrounded values cannot be recovered from those products alone."),
             EvidenceNote(code="ANNOTATION_NOT_INFERRED", message="Secondary-metabolite and occurrence annotations must have "
                          "a declared source and exact source-SMILES identity. They are not inferred from molecular size, "
                          "scaffold or a high model score. Unknown annotation is not a negative annotation. Supplied annotations "
                          "are caller evidence; their botanical correctness is not independently verified by this tool."),
             EvidenceNote(code="RANKING_NOT_EFFICACY", message="The0.70cutoff selects DMPNN-positive records. The0.5evaluation "
                          "threshold is not applied as a final-ranking cutoff. Toxicity predictions are not used in ranking. "
                          "High rank establishes neither efficacy, safety nor independently verified Cannabis occurrence.")]
    notes.extend([
        EvidenceNote(code="RANK_DIRECTION_AND_POPULATION", message="Rank1 is best. Rank change is structural rank minus "
                     "product rank within the SAME complete source query cohort: positive=improved, negative=worsened. "
                     "It is NOT a difference from paper_rank, which belongs to a different selected population. "
                     "Both paper and source ranks are provided for the same mapped compounds; their correspondence "
                     "is available, but different populations prevent interpreting their difference as an improvement."),
        EvidenceNote(code="CROSS_MODEL_SCORE_COMPARISON", message="The author component probabilities are unknown. "
                     "A discrepancy between a reported final score and new structural probability does not establish "
                     "contribution by another author component. Models/cohorts differ. Within one model calculation, "
                     "a product of probabilities in[0,1] cannot exceed either factor or rescue a failed structural screen. "
                     "For the fixed fitted predictors used here, changing the query population can change ranks, "
                     "but does not by itself explain different predictions for the same molecule. Training/model "
                     "differences have not been isolated by a matched comparison. Unreported original component "
                     "scores limit the comparison; their absence is not itself a cause of changed predictions."),
        EvidenceNote(code="CANDIDATE_PARAMETER_PROVENANCE", message="The0.70 candidate-screen cutoff is reported in "
                     "the article and retained in this reconstruction. Scheme, iterations and max_epochs describe "
                     "the executed reconstruction; their presence in metadata does not establish equivalence to "
                     "the author's final fitted models. Do not label every numerical setting as unpublished."),
        EvidenceNote(code="MISSING_ANNOTATION_BLOCKS_SELECTION", message="When annotations are unknown, the requested "
                     "primary-metabolite exclusion cannot be reproduced. This is missing evidence, not an intentional "
                     "change to the requested selection. Zero eligible records with no annotations does not mean all "
                     "records are primary metabolites."),
    ])
    counts = [Metric(name="paper_dmpnn_positive", value=1010, population="final-paper DS1", evidence_kind="paper_reported"),
              Metric(name="paper_secondary_retained", value=44, population="final-paper CCD-filtered candidates", evidence_kind="paper_reported")]
    params = {"reference_only":reference_only, "structure_cutoff":.70, "secondary_annotation_policy":"explicit_only"}
    finding = ("Figure4 names and rounded final products are published references. Original component probabilities "
               "and the final CCD cohort are unavailable; exact author selection is not established.")
    if not reference_only:
        if scheme not in {"random", "scaffold"}:
            raise ValueError("Choose random or scaffold ensemble explicitly")
        structural, runs = fusion_runs("paired_cv", None, max_epochs, iterations)
        directory, info = next((d, i) for d, i in runs if i["scheme"] == scheme)
        scores = pd.read_csv(directory / "query_mean_scores.csv")
        reference_smiles = set(load().loc[load().activity.notna(), "SMILES"].map(canonical_smiles))
        scores["identity_in_labeled_reference"] = scores.SMILES.map(canonical_smiles).isin(reference_smiles)
        ranked = annotate_and_rank(scores, annotations or [])
        components = activity_rows(mapped, ranked)
        subset_metrics, subset_finding = summarize_mapped_references(
            components, source_population=len(ranked), scheme=scheme)
        counts.extend(subset_metrics)
        complete_components = int(ranked[["structure_probability", "rte_probability"]].notna().all(axis=1).sum())
        finding += (f" Both component scores are available for {complete_components}/{len(ranked)} source query records "
                    "before structural-screen and annotation eligibility filters. " + subset_finding)
        unknown = int(ranked.annotation_status.eq("unknown").sum())
        eligible_count = int(ranked.eligible.sum())
        finding += (f" Supplied annotations yield {eligible_count} eligible source records; {unknown} annotations remain "
                    "unknown. The mapped reference list is not a reproduced final source selection.")
        tables["figure4_source_components"] = table(components,
            "New source-model components and ranks joined by exact source ID. Ranks concern all2749source query records before annotations; "
            "rank change=structural rank-product rank (positive means improved position). Not author component probabilities or the full44candidate ranking")
        tables["source_candidate_scores"] = table(ranked, "All2749query records, component scores, preannotation ranks and explicit annotations")
        eligible = ranked.loc[ranked.eligible].sort_values("final_rank_with_supplied_annotations").head(10)
        tables["top10_with_supplied_annotations"] = table(eligible, "Computed on available source cohort only; empty if no verified secondary annotations supplied")
        counts += [metric("source_query_records",len(ranked)), metric("source_dmpnn_positive",int(ranked.structure_screen_positive.sum())),
                   metric("source_unknown_annotations",int((ranked.annotation_status=="unknown").sum())),
                   metric("source_eligible_with_supplied_annotations",int(ranked.eligible.sum())),
                   metric("source_query_identity_in_labeled_reference",int(ranked.identity_in_labeled_reference.sum()))]
        params.update({"scheme":scheme, "max_epochs":max_epochs, "iterations":iterations,
                       "fusion_request_sha256":info["request_sha256"],
                       "execution_stages": {
                           "ranking": "computed_this_call_from_validated_component_scores",
                           "component_score_population": "all source query records before candidate filtering",
                           "component_score_population_records": len(ranked),
                           "records_with_both_component_scores": complete_components,
                           "structural_screen_positive_records": int(ranked.structure_screen_positive.sum()),
                           "product_rank_population_records": len(ranked),
                           "operation_order": ["obtain_validated_component_scores_for_all_source_queries",
                                               "derive_product_scores_and_full_population_ranks",
                                               "apply_structural_screen_and_explicit_annotation_eligibility",
                                               "select_eligible_top10"],
                           "dmpnn_reused_from_cache": structural.metadata.reused_from_cache,
                           "rte_fit_cache_usage": "not_separately_instrumented_by_fusion_backend",
                           "metadata_reused_from_cache_scope": "whole_prioritization_response_only; does_not_describe_component_fits",
                       },
                       "annotations": [CandidateAnnotation.model_validate(v).model_dump() for v in annotations or []]})
        notes.append(EvidenceNote(code="CACHE_FLAG_SCOPE", message="metadata.reused_from_cache=false describes this newly "
                                  "computed prioritization response, not whether component models were fitted afresh. "
                                  "See execution_stages for DMPNN cache evidence; RTE fit cache usage is not separately "
                                  "instrumented. Do not infer no cached fits from the response-level flag."))
        notes.append(EvidenceNote(code="SCORE_TARGET_SCOPE", message="structure_probability and rte_probability "
                                  "are source-activity classification scores from DMPNN and residue-energy models, "
                                  "respectively. Neither is a measured biological effect or receptor-binding/occupancy "
                                  "probability. AChE-inactive source labels do not establish universal absence of "
                                  "pesticidal activity."))
        notes.extend(structural.answer.limitations)
    elif annotations:
        raise ValueError("Annotations require computed source scores; reference_only cannot apply them")
    output = result(finding, counts, tables,
                    "cannabis.candidate_prioritization.v1", notes=notes, **params)
    output.status="partial"
    output.metadata.sources.extend(provenance("cannabis_candidates", "figure4.manual_transcription").sources)
    attach_mapping_provenance(output, base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341"))
    return export_tables(output, port=7341)
