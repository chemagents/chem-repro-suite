"""Article-specific Cannabis MCP; execution and reporting remain in CoScientist."""

from typing import Annotated, Literal

from pydantic import Field

from ..common.contracts import AnalysisResult
from ..common.pool import register, serve, study_server
from . import analysis
from .paper_reference import attach_primary_validation
from .prioritization import CandidateAnnotation

mcp = study_server("Cannabis2026", "10.3390/ijms27188000", artifact_port=7341, operational_numeric_defaults=True)


def dataset_audit() -> AnalysisResult:
    """Which Cannabis source cohorts and six residue-energy matrices are available, and do they
    match the final article? Audit IDs/structures, 3171 labels, 2749 unlabeled source rows and missing
    final 2465/5027 DS1 cohort mapping. AChE-inactive references are not universal non-pesticides.
    Return the separately reported raw/standardized/docking-prepared flow for all three paper cohorts.
    Also return the bundled original paper PDF as a source-reference artifact for CoScientist's existing paper reader.
    """
    return analysis.dataset_audit()


def docking_comparisons() -> AnalysisResult:
    """Do registered pesticides and AChE-inactive references have different docking profiles?
    Recompute six two-sided Mann–Whitney tests, BH FDR, median differences and distributions.
    Does not rerun Vina; tests source scores. Final DS1 subset remains unmapped.
    """
    from .figures import docking
    from .data import load, TARGETS
    return docking(analysis.docking_comparisons(), load(), TARGETS)


def select_residues(
    target: Literal["all", "GSTE2", "ACHE", "GLC1", "UGT202A2", "GSTM12", "OR28"] = "all",
    split_index: Annotated[int, Field(ge=0, le=29)] = 0,
    inner_splits: Annotated[int, Field(ge=2, le=10)] = 5,
    inner_test_size: Annotated[float, Field(ge=0.1, le=0.4)] = 0.2,
    seed: Annotated[int, Field(ge=0, le=2147483647)] = 42,
) -> AnalysisResult:
    """Which target residues survive Appendix A's training-only RMT and 98% energy-contribution gate?
    target=all (default) runs all six targets; select one named target for a narrower diagnostic.
    Fit on an explicit source outer split; choose top-m by inner-validation ROC-AUC on cumulative raw
    energies, retain original columns, export split registry and train/test features. Returns spectrum,
    full m scan and contribution audit. This is not the older global RMT95/point-biserial protocol.
    Unreported numerical choices and unverified final-paper split equivalence are explicitly disclosed.
    """
    from .figures import selection
    if target == "all":
        from ..common.comparison import compare_variants
        from .data import TARGETS
        return compare_variants(
            {name.lower(): selection(analysis.select_residues(name, split_index, inner_splits, inner_test_size, seed))
             for name in TARGETS}, protocol_id="cannabis.appendix_a.all_targets.v1",
            finding="Computed Appendix A diagnostics for every target on the same declared source outer split. "
                    "Each target retains its own eigenspectrum, inner scan, energetic gate and raw residue columns.")
    return selection(analysis.select_residues(target, split_index, inner_splits, inner_test_size, seed))


def validate_rte(
    split_indices: Annotated[
        list[Annotated[int, Field(ge=0, le=29)]] | None, Field(min_length=1, max_length=30)
    ] = None,
    iterations: Annotated[int, Field(ge=10, le=1000)] = 500,
    include_models: bool = False,
    protocol: Literal["paper_kfold10", "source_random"] = "paper_kfold10",
) -> AnalysisResult:
    """Reproduce Figure3 RTE discrimination using CatBoost and linear SVM for all six proteins
    and ALL6. paper_kfold10 (default) follows the caption's ten-fold validation: ten disjoint test
    folds, 140 fits, train-only AppendixA selection and separate fold mean/SD and pooled OOF.
    Source authors' exact memberships and hyperparameters are unavailable; explicit seed42
    stratified folds are saved. source_random preserves all30 deposited repeated80/20holdouts
    (420 fits), a separate code-replay protocol that must not be relabeled ten-fold validation.
    split_indices selects0..9 in paper mode or0..29 in source mode; None runs all folds.
    Return complete predictions, residue-selection audits and scientific figures. For Appendix A
    eigenspectrum and energy-gate plots on a deposited source split, also use select_residues;
    target=all covers all six targets in one response. Verified
    cached training is labeled separately from a new fit. No parameter is tuned to a paper number.
    """
    if protocol == "paper_kfold10":
        from .paper_cv import validate_paper_cv
        evaluated = validate_paper_cv(split_indices, iterations=iterations, include_models=include_models)
    elif protocol == "source_random":
        from .cached_validation import validate_rte_cached
        evaluated = validate_rte_cached(split_indices, iterations=iterations, include_models=include_models)
    else:
        raise ValueError("Unknown RTE validation protocol")
    return attach_primary_validation(evaluated, {"RTE_ALL6"})


for fn in [dataset_audit, docking_comparisons, select_residues, validate_rte]:
    register(
        mcp,
        fn,
        name="cannabis2026_" + fn.__name__,
        dataset_scope="cannabis_2026",
        read_only=fn not in {select_residues, validate_rte},
    )


def open_toxicity_validation(
    endpoint: Literal["all", "ames", "reproductive", "daphnia", "fathead"] = "all",
    scheme: Literal["reported", "random", "source_scaffold"] = "reported",
    cohort: Literal["both", "toxric", "toxric_ecotox"] = "both",
) -> AnalysisResult:
    """Reproduce the complete Supplement S3 open benchmark. Defaults execute every reported
    endpoint/split combination and BOTH explicitly labeled aquatic cohorts (11 replays covering
    seven reported references); no reproductive scaffold value is invented. Choose an explicit
    endpoint/scheme/cohort for one replay. All five base learners, mean blend and source meta-stack
    are retained, with actual saved OOF scores independently verified. No estimator is selected
    to force agreement. Return a combined table and same-cohort/same-estimator random/scaffold
    comparisons; source meta CV is not strictly nested or scaffold-disjoint. Open benchmarks are
    separate from main-text Syntelly profiles. Requires the pinned open-toxicity worker environment.
    """
    from .toxicity_benchmarks import evaluate_benchmarks
    return evaluate_benchmarks(endpoint, scheme, cohort)


register(
    mcp,
    open_toxicity_validation,
    name="cannabis2026_open_toxicity_validation",
    dataset_scope="cannabis_open_tox_2026",
    read_only=False,
)


def open_toxicity_profile(
    endpoints: Annotated[list[Literal["ames", "reproductive", "daphnia", "fathead", "dili", "herg",
                                     "carcinogens_lagunin", "mouse_oral_ld50", "rat_oral_ld50"]] | None,
                         Field(min_length=1, max_length=9)] = None,
    recipe: Literal["fixed_public_v1", "nested_paper2024_v2"] = "fixed_public_v1",
) -> AnalysisResult:
    """How do the available Cannabis compounds compare with pesticide references under open
    toxicity models, and how applicable are the predictions? Fit the existing ChemAgents open
    two-booster approximation motivated by Syntelly 2024, validate identity-random/scaffold 5-fold,
    and predict every source compound. Return per-compound values, cohort comparisons, kNN domain
    scores and training overlap. Uses verified caches after the first CPU run. These are new open
    predictions; final-paper Syntelly weights/cohorts and absolute aquatic units are unavailable.
    Includes named predictions and endpoint-specific domain/overlap matrices for all ten reviewed
    Figure4 connectivities, with full candidate-level CSV evidence. These are source-model estimates.
    hERG is channel blockade; a separate broad hepatotoxicity output is not invented from DILI.
    recipe=nested_paper2024_v2 replays completed nested-tuning runs with applied 2024 preparation,
    inner-fold feature selection and verified saved models. Returns each requested endpoint in a coverage table; absent completed models are explicitly
    unavailable while other endpoints are still verified. No fallback to fixed-v1 predictions.
    """
    from .figures import toxicity
    if recipe == "nested_paper2024_v2":
        from .nested_toxicity import evaluate

        return toxicity(evaluate(endpoints))
    if recipe != "fixed_public_v1":
        raise ValueError("Unknown public toxicity recipe")
    from .toxicity_profiles import open_toxicity_profile as evaluate

    return toxicity(evaluate(endpoints))


register(mcp, open_toxicity_profile, name="cannabis2026_open_toxicity_profile",
         dataset_scope="cannabis_2026", read_only=False)


def validate_dmpnn(
    protocol: Literal["paired_cv", "source_random"] = "paired_cv",
    source_split_indices: Annotated[list[Annotated[int, Field(ge=0, le=29)]] | None,
                                    Field(min_length=1, max_length=30)] = None,
    max_epochs: Annotated[int, Field(ge=20, le=200)] = 60,
    include_models: bool = False,
) -> AnalysisResult:
    """Does the structure-based Cannabis DMPNN transfer to unseen scaffolds? Actually train Chemprop
    directed-bond message passing plus train-normalized RDKit2D descriptors under the published
    depth3/hidden300/dropout0.1/mean-sum-norm configuration. paired_cv evaluates both random and
    scaffold five-fold CV, with separate validation subsets, best checkpoints and complete held-out
    predictions. source_random uses the existing RTE outer split registry for matched fusion studies;
    it defaults to all30source splits, or the explicitly requested indices. Original final3403labels
    and author fold/checkpoint details are absent: the available3171labels are used and all deviations
    declared. PR trapezoidal area and average precision are separate; class scores are not measured
    pesticide efficacy. The reported3403 standardized reference molecules are not proof of the
    authors' exact DMPNN training membership. Long CPU training is cached; incomplete runs resume only completed verified folds.
    """
    from .structure_validation import validate_structure

    return attach_primary_validation(validate_structure(protocol, source_split_indices, max_epochs, include_models), {"DMPNN"})


register(mcp, validate_dmpnn, name="cannabis2026_validate_dmpnn", dataset_scope="cannabis_2026", read_only=False)


def evaluate_fusion(
    protocol: Literal["paired_cv", "source_random"] = "paired_cv",
    source_split_indices: Annotated[list[Annotated[int, Field(ge=0, le=29)]] | None,
                                    Field(min_length=1, max_length=30)] = None,
    max_epochs: Annotated[int, Field(ge=20, le=200)] = 60,
    iterations: Annotated[int, Field(ge=10, le=1000)] = 500,
    include_models: bool = False,
) -> AnalysisResult:
    """Does multiplying Cannabis DMPNN and RTE scores improve held-out classification?
    Train an ALL6 CatBoost branch with AppendixA train-only residue selection on exactly the
    structural outer partitions. Compare both branches and product at threshold0.5: ROC-AUC,
    both PR-area definitions, FPR, recall, confusion counts and positive retention. Return all
    matched held-out/query predictions and optional model exports. Reuse verified DMPNN fits.
    Product is not a calibrated joint posterior; lower FPR may accompany lower recall. Available
    source cohort and reconstructed folds are not claimed to reproduce author final checkpoints.
    """
    from .fusion import evaluate_fusion as evaluate
    from .fusion_summary import attach_fusion_summary

    from .figures import fusion
    return attach_primary_validation(fusion(attach_fusion_summary(
        evaluate(protocol, source_split_indices, max_epochs, iterations, include_models))), {"DMPNN", "RTE_ALL6", "product"})


register(mcp, evaluate_fusion, name="cannabis2026_evaluate_fusion", dataset_scope="cannabis_2026", read_only=False)


def prioritize_candidates(
    annotations: list[CandidateAnnotation] | None = None,
    reference_only: bool = False,
    scheme: Literal["random", "scaffold"] = "random",
    max_epochs: Annotated[int, Field(ge=20, le=200)] = 60,
    iterations: Annotated[int, Field(ge=10, le=1000)] = 500,
) -> AnalysisResult:
    """Which Cannabis candidates rank highest, and can Figure4 be reproduced?
    Return the published top10 names/rounded product scores separately from new source-cohort
    probabilities and ranks. Computed mode uses cached paired DMPNN/RTE ensembles and the0.70
    structural screen. Secondary-metabolite/occurrence annotations require an explicit source and
    matching SMILES; missing annotation stays unknown and is never silently excluded as primary.
    No0.5final-ranking cutoff or toxicity selection is applied. reference_only skips model fits.
    Reviewed Figure4 graphs are linked to exact source records: return their identity evidence and,
    in computed mode, newly reconstructed component probabilities separately from reported scores.
    Without author final CCD cohort/annotations, original component scores and exact final selection
    remain unavailable; no names, biological occurrence or separate probabilities are invented.
    """
    from .prioritization import prioritize_candidates as prioritize

    from .figures import candidates
    return candidates(prioritize(annotations, reference_only=reference_only, scheme=scheme,
                      max_epochs=max_epochs, iterations=iterations))


register(mcp, prioritize_candidates, name="cannabis2026_prioritize_candidates",
         dataset_scope="cannabis_2026", read_only=False)


def main():
    serve(mcp, default_port=7341)


if __name__ == "__main__":
    main()
