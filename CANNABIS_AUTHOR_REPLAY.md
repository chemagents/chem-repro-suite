# Reproducing Cannabis from the deposited code

The implementation is pinned to [canpest b43286cc](https://github.com/chemagents/canpest/tree/b43286cc799f5ee856e4ed5d546078ce8c26e270). Its five Python files are preserved byte for byte in `chem_repro/vendor/canpest_author`, with source hashes. The adapter changes paths and assembles the missing training loop; it calls the original training, selection and veto functions.

Call `cannabis2026_reproduce_author_code` to recompute metrics from complete held-out predictions of author-code refits. Every test ID, source label and product probability is checked against the deposited split registry. Outputs include checkpoint-reload receipts, confusion counts, recall, false positives removed, true positives lost, and a plot. These are new fitted weights, not recovered historical checkpoints.

| Item | Deposited implementation | Article / supplement |
|---|---|---|
| DMPNN | Custom PyTorch encoder; dropout 0.15; mean, sum/100 and max | Chemprop; dropout 0.1; mean, sum and norm |
| Validation | Thirty fixed repeated 80/20 outer splits | Figure 3 specifies ten-fold validation for RTE models |
| Residue selection | Global-label point-biserial/RMT ranking; 45 GLC1 + 50 ACHE | Appendix A: training-only selection and energy-contribution gate |
| Inner diagnostic | LR fitted and evaluated on the same validation scores | Inner validation to choose retained residues |
| Diagnostic SVM | RBF | Linear |
| Combination | DMPNN × CatBoost(RMT95) | Deposited documentation attributes final numbers to a later RMT95+RTE390 geometric mix; article §4.7 gives a product rule |
| Reference FPR | 5.14% | 4.92% |
| Reference ROC-AUC | 0.9167 | 0.9176 |

The replay preserves source behavior, including global-label selection and its in-sample inner diagnostic. It demonstrates execution fidelity, not leakage-free independent validation. Existing `validate_rte`, `validate_dmpnn` and `evaluate_fusion` tools retain their separately described article-based reconstructions. Never select a protocol or hyperparameter because its score is closer to the paper.

## Refit the original code

Replay needs only the base environment. Refitting also needs the pinned PyTorch worker dependencies. Prepare empty project and results directories:

```bash
python scripts/setup_cannabis_author_replay.py \
  --project-dir /absolute/path/cannabis-source-run \
  --output-dir /absolute/path/cannabis-source-results
```

Create a Python 3.12 environment using the generated `requirements.txt`. Commit the generated project and execute its fixed `python -u replay.py` command through your compute runner. The configuration runs all30 original splits, recomputes95 selected features, trains the original DMPNN/descriptor model and HGB baseline, applies the source validation-selected blend, reloads checkpoints, and executes the original CatBoost/SVM veto. Metadata records whether the source score called DMPNN has a DMPNN/HGB blend weight exactly equal to one.

Outputs include predictions, per-split metrics, means and sample SDs, weights, reload checks, input/source hashes and effective configuration. Incomplete runs are not replaced with published reference values.

## Recovered local author table

`cannabis2026_recovered_author_cohort` audits the local `Docking and MIE/Activity/victor.csv`:2465 unique IDs with exact matching deposited SMILES;1010 archived `precision` values >=0.70. All inclusions,284 remaining source-query IDs and all ten Figure4 structure matches are exported.

The model producing `precision` is unresolved. Do not automatically call it the initial DMPNN probability or final product. Chemical class fields are retained as recorded, not converted into CCD secondary-metabolite labels. Count agreement alone does not recover the44-compound selection or validate annotations. The report must retain cohort, score origin and protocol beside each result. Original Syntelly outputs remain source references unless separately recomputed with the corresponding data and model.
