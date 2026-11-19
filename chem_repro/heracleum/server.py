"""Heracleum 2026 scientific adapter over the shared chemistry tool pool."""

from typing import Annotated, Literal

from pydantic import Field

from ..common.artifacts import export_tables
from ..common.contracts import AnalysisResult, IdentityComparisonPolicy, IdentityPolicy
from ..common.pool import register, serve, study_server
from . import analysis, docking

mcp = study_server("Heracleum2026", "10.3390/ijms27177947", artifact_port=7342, operational_numeric_defaults=True)


def dataset_audit() -> AnalysisResult:
    """What are the DS1/DS2 sizes, overlap, archived docking coverage and source identity conflicts?
    Compare master versus source canonical SMILES by synID and export problematic rows separately.
    Unusable docking profiles mean all-zero archived scores, not geometry/valence errors; all molecular
    graphs in this audit parse. No new docking, conformer validation or SMILES-versus-InChI test is run.
    """
    return export_tables(analysis.dataset_audit(), port=7342)


def chemical_space(include_figures: bool = True) -> AnalysisResult:
    """How concentrated is Heracleum furanocoumarin chemistry around psoralen and angelicin?
    Recompute both source datasets' Murcko/generic counts, top-2/5/15 coverage and 50% coverage.
    """
    result = analysis.chemical_space()
    if include_figures:
        from .figures import chemical_space as plot

        result = plot(result)
    return export_tables(result, port=7342)


def docking_profile(include_figures: bool = True) -> AnalysisResult:
    """Does averaging 44 docking scores hide target-profile information? Recompute PCA, loadings,
    explained variance, PC1–mean correlation and strong-score counts on the deposited 2055 profiles.
    """
    result = analysis.docking_profile()
    if include_figures:
        from .figures import docking_profile as plot

        result = plot(result)
    return export_tables(result, port=7342)


def docking_protocol_validation(
    mode: Literal["published_table", "public_reconstruction"] = "published_table",
    include_structures: bool = True,
) -> AnalysisResult:
    """How reliable is the Heracleum docking protocol? Recompute top-1 through top-5 recovery
    from the 44 published Table S1a target observations, with 42 evaluated targets and two exclusions.
    Default published_table aggregates source observations without rerunning docking.
    public_reconstruction runs the declared public-crystal Vina protocol or verifies an existing
    completed run, recalculates symmetry-aware heavy-atom RMSD without superposition, and exports
    structures and preparation failures. Original prepared author receptors and raw active/inactive
    controls remain unavailable; reconstructed outcomes are never labeled exact author-input replay.
    """
    if mode not in {"published_table", "public_reconstruction"}:
        raise ValueError("Unknown docking validation mode")
    from .control_reference import attach_control_reference

    if mode == "public_reconstruction":
        from .redocking import reconstruct

        return export_tables(attach_control_reference(reconstruct(include_structures)), port=7342)
    return export_tables(attach_control_reference(docking.protocol_validation()), port=7342)


def toxicity_landscape(include_figures: bool = True) -> AnalysisResult:
    """Can ten predicted Heracleum toxicities be reduced to one index? Recompute distributions,
    saturation, Spearman correlations and standardized PCA of the deposited ADMETlab values.
    """
    result = analysis.toxicity_landscape()
    if include_figures:
        from .figures import toxicity_landscape as plot

        result = plot(result)
    return export_tables(result, port=7342)


def scaffold_separability(
    permutations: Annotated[int, Field(ge=99, le=10000)] = 10000, include_figures: bool = True
) -> AnalysisResult:
    """Are five scaffold classes separated beyond chance across ten toxicity and three docking-PC axes?
    Run mean-pairwise KS permutations and BH FDR. 10000 is the original protocol; Other is excluded.
    """
    result = analysis.separability(permutations)
    if include_figures:
        from .figures import separability as plot

        result = plot(result)
    return export_tables(result, port=7342)


def scaffold_transfer(
    endpoint: Literal["both", "neurotoxicity", "eye_corrosion"] = "both",
    model: Literal["both", "RF", "ElasticNet"] = "both",
    identity_policy: IdentityPolicy = "strict",
    bootstrap: Annotated[int, Field(ge=100, le=1000)] = 1000,
    include_oof: bool = True,
    include_figures: bool = True,
) -> AnalysisResult:
    """Does a Morgan-fingerprint surrogate transfer to unseen generic scaffolds? Refit RF/ElasticNet
    with random and GroupKFold CV, OOF metrics and CI. Default both/both completes ALL FOUR model ×
    endpoint combinations and both split schemes in one scientific call; individual selections remain
    supported. No separate calls are needed for the full comparison. Published snapshot reproduces notebook inputs;
    strict excludes identity conflicts. Model targets are ADMETlab predictions, not experimental labels.
    eye_corrosion is distinct from eye_irritation. Compare absolute scaffold performance together with
    the random-minus-scaffold R² drop; a smaller drop alone does not establish a more robust model.
    """
    from .transfer_grid import compare_transfer

    result = compare_transfer(endpoint, model, identity_policy, bootstrap, include_oof, include_figures)
    return export_tables(result, port=7342)


def prioritize_candidates(
    identity_policy: IdentityComparisonPolicy = "strict",
    strong_threshold: Annotated[float, Field(ge=-30, lt=0, allow_inf_nan=False)] = -10,
    include_figures: bool = True,
) -> AnalysisResult:
    """Which natural Heracleum compounds merit follow-up and why? Apply the original eight-endpoint,
    docking, PAINS and Brenk rules. Strict leaves identity conflicts unranked; published_snapshot reproduces
    the deposited tier counts despite conflicts. All 102 records remain the median reference in both modes.
    Compare Tier 1 with published Table S4 separately by original DS2 and master prediction structures;
    report named-priority claim disagreements rather than repeating them as computed findings.
    Use identity_policy="compare" for ONE call containing BOTH snapshot and strict lists with their
    separate criteria, identity diagnostics and downloadable files. No second output slot is needed.
    """
    if identity_policy == "compare":
        from ..common.comparison import compare_variants, compare_record_tables
        from ..common.contracts import Table
        variants = {mode: analysis.prioritize(mode, strong_threshold)
                    for mode in ("published_snapshot", "strict")}
        comparison = compare_record_tables(
            variants["published_snapshot"].answer.tables["candidates"],
            variants["strict"].answer.tables["candidates"], id_column="synID",
            fields=["tier", "identity_verified", "n_favorable", "n_strong", "pains", "brenk"],
            reference_label="published_snapshot", observed_label="strict")
        changes = [row for row in comparison.rows if "tier" in row["changed_fields"]]
        priority_changes = [row for row in changes if row["published_snapshot__tier"] == 1]
        summaries = []
        for mode, variant in variants.items():
            rows = variant.answer.tables["candidates"].rows
            summaries.append({
                "identity_policy": mode,
                **{f"tier_{tier}": sum(row["tier"] == tier for row in rows) for tier in (1, 2, 3)},
                "unranked": sum(row["tier"] is None for row in rows),
                "ranked": sum(row["tier"] is not None for row in rows),
                "total_records": len(rows),
            })
        if include_figures:
            from .figures import candidate_structures
            variants = {mode: candidate_structures(value) for mode, value in variants.items()}
        result = compare_variants(
            variants,
            protocol_id="heracleum.prioritization.identity_comparison.v2",
            finding="Published-snapshot and strict prioritization are computed separately on the same 102-record reference. "
                    "Conflicting identities are retained in the snapshot and left unranked in strict mode. "
                    "The tiers prioritize experiments; they do not establish safety or biological validation.",
        )
        result.answer.tables["identity_policy_record_comparison"] = comparison
        result.answer.tables["identity_policy_summary"] = Table(
            description="Counts recomputed from each complete candidate table; unranked records remain in the total",
            columns=list(summaries[0]), rows=summaries)
        result.answer.tables["changed_records"] = Table(
            description="Every record whose tier changes between the two identity policies",
            columns=comparison.columns, rows=changes)
        result.answer.tables["changed_snapshot_tier1_records"] = Table(
            description="Published-snapshot Tier 1 records affected by identity verification; IDs are not verified names",
            columns=comparison.columns, rows=priority_changes)
        population = "same DS2 records matched by unique synID; no identity repair"
        result.answer.metrics.extend([
            analysis.metric("records_compared_by_synid", len(comparison.rows), population=population),
            analysis.metric("records_with_tier_change", len(changes), population=population),
            analysis.metric("snapshot_tier1_records_with_tier_change", len(priority_changes), population=population),
            analysis.metric("records_without_tier_change", len(comparison.rows)-len(changes), population=population),
        ])
        result.answer.finding += (
            " " + "; ".join(
                f"{row['identity_policy']}: Tier 1/2/3 = {row['tier_1']}/{row['tier_2']}/{row['tier_3']}, "
                f"{row['ranked']} ranked, {row['unranked']} unranked, {row['total_records']} total"
                for row in summaries) + "."
            f" Full synID-matched comparison covers {len(comparison.rows)} records: "
            f"{len(changes)} change tier, including {len(priority_changes)} snapshot Tier 1 records. "
            "The complete comparison and affected Tier 1 records are computed tables, not an inference from previews.")
        return export_tables(result, port=7342)
    result = analysis.prioritize(identity_policy, strong_threshold)
    if include_figures:
        from .figures import candidate_structures

        result = candidate_structures(result)
    return export_tables(result, port=7342)


for fn in [
    dataset_audit,
    chemical_space,
    docking_profile,
    docking_protocol_validation,
    toxicity_landscape,
    scaffold_separability,
    scaffold_transfer,
    prioritize_candidates,
]:
    register(
        mcp,
        fn,
        name="heracleum2026_" + fn.__name__,
        dataset_scope="heracleum_2026",
        read_only=fn == dataset_audit,
    )


def main():
    serve(mcp, default_port=7342)


if __name__ == "__main__":
    main()
