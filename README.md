# ChemAgents scientific MCP runtime

Shared chemistry, statistics, artifact export and paper-specific MCP implementations for ITMO.CoScientist. The three server repositories use an immutable commit of this package, so scientific code is maintained in one place.

| Server | Paper | Tools | Default URL |
|---|---|---:|---|
| Heracleum | IJMS 2026, 27, 7947 | 8 | http://127.0.0.1:7342/mcp |
| Mannich | Scientific Data s41597-026-08287-9 | 10 | http://127.0.0.1:7343/mcp |
| Cannabis | IJMS 2026, 27, 8000 | 9 | http://127.0.0.1:7341/mcp |

## Install and run

Python 3.12 and uv are required. The base servers do not require an LLM API key.

```bash
uv sync --frozen --python 3.12
uv run heracleum-2026-mcp
# or, in another terminal:
uv run mannich-2026-mcp
```

Use Streamable HTTP at `/mcp` for the complete tool and downloadable-artifact workflow. Set `CHEM_REPRO_ARTIFACTS_DIR` to a persistent output directory and `CHEM_REPRO_PUBLIC_BASE_URL` to a URL reachable by the client when hosting elsewhere. Artifacts include their size and SHA-256.

For Cannabis, follow [CANNABIS_DEPLOYMENT.md](CANNABIS_DEPLOYMENT.md): neural and toxicology computations use separately pinned worker environments. Dataset auditing works with the base environment; model training needs the workers. Missing author models or unsupported modes are reported explicitly.

## Integration and validation

Register `rag_registration.json` through CoScientist's existing MCP registry, with URLs adjusted to your deployment. CoScientist performs routing, planning, code execution and reporting. These servers provide scientific operations and structured evidence; they do not duplicate that orchestration.

```bash
uv run pytest tests
```

The public suite tests scientific logic and MCP contracts. Twenty-eight natural-language system prompts and separate expected checks are under `tests/e2e`. The separate system evaluation is ongoing: successful calculations and direct MCP checks do not certify a correct final LLM report. Known CoScientist orchestration/reporting issues and compatibility patches are handled separately in its runtime.

## Scientific boundaries

- Heracleum: `strict` is the default identity policy. Use `published_snapshot` for the deposited numerical population. There are 65 master/source identity conflicts. The tool preserves them and separately compares Tier1 with Supplementary TableS4. Pabularinone scores 7/8 favorable endpoints in the archived data, contrary to the paper's 8/8 statement. Predictions are ADMETlab outputs, not measured toxicity. Published docking/control summaries and a new public-structure reconstruction are separate evidence types.
- Mannich: the 4015 reaction records are already manually curated; no reaction OCR or re-extraction is implemented. Axial identities, duplicates with different outcomes and uncertain geometry remain explicit. The 207 versus 204 catalyst-scaffold discrepancy and unavailable exact reaction-PCA settings are unresolved.
- Cannabis: the public source cohort and newly fitted models are not silently substituted for unavailable author cohorts/weights. Final CCD annotations and original Syntelly data/weights are missing. Public reconstruction and nested evaluation are labelled accordingly; the nested recipe requires verified completed results and is not configured by the basic worker installer.

See [REPRODUCTION_REQUIREMENTS.md](REPRODUCTION_REQUIREMENTS.md), [MISSING_SOURCE_INPUTS.md](MISSING_SOURCE_INPUTS.md) and [NOTICE.md](NOTICE.md). These limitations are returned with results rather than hidden by changing the data or acceptance criteria.
