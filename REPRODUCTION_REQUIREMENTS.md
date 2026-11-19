# Scientific coverage and validation scope

The servers expose29 fixed-dataset scientific operations:8 Heracleum,10 Mannich and11 Cannabis. Shared code implements chemistry, statistics, validation, provenance and artifact export. CoScientist retains orchestration and report writing.

## Heracleum

Implemented: source identity audit; scaffold diversity; docking PCA and loadings; published pose-recovery table aggregation; public-structure redocking; toxicity distributions/correlations/PCA; 13-axis permutation analysis; RF/ElasticNet comparison for both endpoints and both splitting schemes; candidate priorities and Table S4 identity/claim checks.

Numerical reproduction uses the explicitly selected `published_snapshot` population. The default `strict` policy preserves identity constraints. Original docking preparation and raw active/inactive control observations remain unavailable. A fresh public reconstruction is a separate result.

## Mannich

Implemented: overview and record audit; ee-to-energy conversion; reaction-condition statistics; priority-ordered SMARTS classification; component scaffolds; selectivity heatmaps; chemical-space embeddings; point/axial stereochemistry checks; regeneration from existing SMILES; declared reaction-PCA reconstruction.

The 4015 records are already manually curated. OCR and re-extraction from primary papers are excluded. Uncertain stereochemistry and the 207-versus-204 scaffold discrepancy are retained.

## Cannabis

Implemented: source cohort audit; docking comparisons; RMT/RTE analysis; CatBoost/SVM validation; Chemprop structural-model validation; paired fusion; candidate filtering/ranking; open toxicology reconstruction and applicability analysis; published-reference comparison.

The deposited-code replay additionally uses the original canpest DMPNN/RMT95 functions and30fixed outer splits, independently verifies checkpoints, and recomputes all held-out metrics. The local2465-record archive is aligned separately, with score provenance explicitly unresolved. See [the protocol comparison](CANNABIS_AUTHOR_REPLAY.md).

Neural and toxicology workers need separate environments. Missing original cohorts, annotations, weights and historical validation details are not filled with guessed values. Nested reconstruction consumes verified completed runs; the basic installer configures the fixed-public recipe.

## Checks

The public unit suite passed 137 tests in a fresh Python 3.12 environment. It includes small mocked control-flow fixtures as well as chemistry/data checks; mocked fixtures are never production scientific results. Full computational checks and natural-language CoScientist runs are distinct validation layers.

The 28 prompts in `tests/e2e/prompts.jsonl` are system-level requests. Their numerical expectations are kept separately in `cases.json` and must not be injected into model prompts. Full qualitative and quantitative reproduction of all three papers is not certified: source limitations and remaining agent/report failures are explicit.
