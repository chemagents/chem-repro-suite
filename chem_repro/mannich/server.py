"""Mannich article MCP; starts after manually extracted reaction records are provided."""

from typing import Literal

from ..common.contracts import AnalysisResult
from ..common.pool import register, serve, study_server
from . import analysis

mcp = study_server("Mannich2026", "10.1038/s41597-026-08287-9", artifact_port=7343, operational_numeric_defaults=True)


def dataset_overview(include_audit: bool = True) -> AnalysisResult:
    """How many asymmetric organocatalytic Mannich reactions, source papers, catalysts and substrates
    are in the curated dataset? Recompute identities including R/S axial configurations; audit
    missingness, units/ranges, dr ratios, duplicate inputs, charge/heavy-atom inventories and unspecified
    product carbon stereo. Full row diagnostics are downloadable; no automatic exclusions or OCR.
    """
    return analysis.overview(include_audit=include_audit)


def enantioselectivity_distribution(include_figures: bool = True) -> AnalysisResult:
    """Why convert Mannich ee to |ΔΔG‡|? Recompute the Eyring conversion, target distribution and
    skewness; check stored values, temperature units and the 99.9% cap. Never conflate ee with dr.
    """
    result = analysis.target_distribution()
    if include_figures:
        from .figures import target_distribution

        return target_distribution(result)
    return result


def reaction_conditions(include_figures: bool = True) -> AnalysisResult:
    """What temperatures, solvents, times, yields and catalyst loadings occur in the Mannich data?
    Recompute Figure 3 statistics with declared mixture/solvent normalization and missingness.
    """
    result = analysis.conditions()
    if include_figures:
        from .figures import reaction_conditions as plot_conditions

        return plot_conditions(result)
    return result


def component_classification(include_figures: bool = True) -> AnalysisResult:
    """Which catalyst, nucleophile and imine families dominate asymmetric Mannich reactions?
    Recompute the paper's ordered SMARTS rules; preserve distinct axial catalysts and count both
    unique structures and reaction records. Audit disagreements with deposited catalyst labels.
    """
    result = analysis.component_classes()
    if include_figures:
        from .figures import representative_structures

        return representative_structures(result)
    return result


def scaffold_diversity(include_figures: bool = True) -> AnalysisResult:
    """How many Murcko scaffolds are represented by Mannich catalysts and substrates?
    Recompute the three supplementary scaffold populations; distinguish acyclic exclusions,
    reaction frequency and unique chemical identities. Axial identity is preserved before grouping.
    """
    result = analysis.scaffold_diversity()
    if include_figures:
        from .figures import scaffold_structures

        return scaffold_structures(result)
    return result


def selectivity_heatmaps(
    protocol: Literal["paper", "notebook"] = "paper", include_figures: bool = True
) -> AnalysisResult:
    """Which Mannich catalyst–substrate classes have high median ee and adequate observation counts?
    Recompute both Figure 7 tables. Paper masks n<5; notebook uses n<4. These are descriptive
    literature medians, not causal catalyst recommendations or prospective validation.
    """
    result = analysis.selectivity_heatmaps(protocol)
    if include_figures:
        from .figures import selectivity_heatmaps as plot_heatmaps

        return plot_heatmaps(result)
    return result


def chemical_space(
    population: Literal["catalysts", "reagents"] = "catalysts",
    protocol: Literal["notebook", "chirality_sensitivity"] = "notebook",
    include_figures: bool = True,
) -> AnalysisResult:
    """How are the 468 Mannich catalysts and substrate pools distributed in chemical space?
    Recompute Figures 4/6 Morgan t-SNE coordinates, source identities and class labels.
    Notebook catalyst fingerprints omit chirality despite the paper wording; a separately labeled
    point-chirality sensitivity mode is available. Axial R/S is not encoded in Morgan bits.
    """
    from .space import chemical_space as compute_space

    result = compute_space(population, protocol)
    if include_figures:
        from .figures import chemical_space as plot_space

        return plot_space(result, population)
    return result


def catalyst_structure_audit() -> AnalysisResult:
    """Are the Mannich catalyst SDF files consistent with the curated SMILES and axial R/S annotations?
    Audit all actual SDF records, explicit point stereochemistry, finite 3-D geometry and axial CIP
    evidence. Exclude macOS archive metadata; do not confuse an R/S text property with verified geometry.
    """
    from .stereochemistry import audit_catalyst_structures

    return audit_catalyst_structures()


def reaction_space(include_agents: bool = True, include_figures: bool = True) -> AnalysisResult:
    """Reconstruct the supplementary Mannich reaction PCA for all curated records. Roles are
    reactants=nucleophile+electrophile, agent=catalyst, product=product. Export coordinates,
    structural fingerprint bits and PCA loadings. The supplement omits exact fingerprint/PCA
    settings; this declared reconstruction is not asserted to reproduce its unknown coordinates.
    No OCR, atom mapping or target-informed feature construction is performed.
    """
    from .reaction_space import coordinates
    from .reaction_space import reaction_space as calculate

    result = calculate(include_agents)
    if include_figures:
        from .figures import reaction_pca

        return reaction_pca(result, coordinates(include_agents)[0])
    return result


def regenerate_catalyst_structures(
    protocol: Literal["source_notebook", "geometry_verified"] = "source_notebook",
    filenames: list[str] | None = None,
) -> AnalysisResult:
    """Regenerate Mannich catalyst SDFs from the curated SMILES and axial labels with the pinned
    source ETKDGv3/MMFF94 algorithm (five conformers,20 bridged seeds,seed42). Return per-record
    failures, point/axial checks, aligned RMSD to deposited structures and a downloadable SDF archive.
    geometry_verified is a separate corrected protocol using nominated-axis CIP constraints; it never
    masquerades as source replay. Default covers all468; filenames optionally selects exact source names.
    """
    from .conformers import regenerate_catalysts

    return regenerate_catalysts(protocol, filenames)


for fn in [
    dataset_overview,
    enantioselectivity_distribution,
    reaction_conditions,
    component_classification,
    scaffold_diversity,
    selectivity_heatmaps,
    chemical_space,
    catalyst_structure_audit,
    reaction_space,
    regenerate_catalyst_structures,
]:
    register(
        mcp,
        fn,
        name="mannich_" + fn.__name__,
        dataset_scope="mannich_2026",
        read_only=fn
        not in {
            dataset_overview,
            enantioselectivity_distribution,
            reaction_conditions,
            selectivity_heatmaps,
            chemical_space,
            component_classification,
            scaffold_diversity,
            reaction_space,
            regenerate_catalyst_structures,
        },
    )


def main():
    serve(mcp, default_port=7343)


if __name__ == "__main__":
    main()
