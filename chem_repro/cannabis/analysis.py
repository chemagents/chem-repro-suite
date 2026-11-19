"""Source audit, six-target statistics and final-paper residue selection."""

import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import mannwhitneyu

from ..common.artifacts import save_artifact, save_json
from ..common.chemistry import canonical_smiles, scaffold
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table
from ..common.rmt import fit_residue_selector
from ..common.statistics import bh_fdr, distribution
from .data import TARGETS, limitations, load, residue_matrix, split


def result(finding, metrics, tables, protocol, notes=(), status="partial", **parameters):
    return AnalysisResult(
        status=status,
        answer=Answer(
            finding=finding, metrics=metrics, tables=tables, limitations=limitations() + list(notes)
        ),
        metadata=provenance("cannabis", protocol, **parameters),
    )


def metric(name, value, population="deposited data.csv", unit=None):
    return Metric(name=name, value=value, population=population, unit=unit)


def dataset_audit():
    df = load()
    paper_sources = provenance("cannabis_candidates", "paper.cohort_flow.transcription")
    paper_flow = pd.read_csv(DATA_ROOT / "cannabis_candidates/paper_cohort_flow.csv")
    paper_totals = paper_flow.groupby("paper_stage").paper_n.sum()
    arithmetic = pd.DataFrame([{
        "standardized_total": int(paper_totals['standardized']),
        "docking_prepared_total": int(paper_totals['docking_prepared']),
        "excluded_by_subtraction": int(paper_totals['standardized']-paper_totals['docking_prepared']),
        "paper_reported_excluded_section_4_4": 2704,
        "difference": int(paper_totals['standardized']-paper_totals['docking_prepared'])-2704,
    }])
    rows = []
    identity_maps = {}
    for cohort, mask in [
        ("DS2_pesticides", df.activity == 1),
        ("DS3_AChE_inactive", df.activity == 0),
        ("source_unlabeled", df.activity.isna()),
    ]:
        selected = df.loc[mask]
        identities = []
        invalid = []
        exact, nonstereo = defaultdict(list), defaultdict(list)
        for row in selected.itertuples():
            try:
                identity = canonical_smiles(row.SMILES)
                identities.append(identity)
                exact[identity].append(int(row.ligand_id))
                mol = Chem.MolFromSmiles(identity)
                Chem.RemoveStereochemistry(mol)
                nonstereo[Chem.MolToSmiles(mol, isomericSmiles=True)].append(int(row.ligand_id))
            except ValueError:
                invalid.append(int(row.ligand_id))
        identity_maps[cohort] = {"exact_isomeric": exact, "stereo_ignored_isotopes_retained": nonstereo}
        rows.append(
            {
                "cohort": cohort,
                "n_records": len(selected),
                "unique_isomeric_structures": len(set(identities)),
                "duplicate_structure_records": len(identities) - len(set(identities)),
                "invalid_smiles_ligand_ids": invalid,
            }
        )
    coverage = []
    overlap_summary, overlap_rows = [], []
    for left, right in combinations(identity_maps, 2):
        for policy in ["exact_isomeric", "stereo_ignored_isotopes_retained"]:
            a, b = identity_maps[left][policy], identity_maps[right][policy]
            shared = sorted(set(a) & set(b))
            overlap_summary.append(
                {
                    "left_cohort": left,
                    "right_cohort": right,
                    "identity_policy": policy,
                    "shared_identities": len(shared),
                    "left_records": sum(len(a[s]) for s in shared),
                    "right_records": sum(len(b[s]) for s in shared),
                }
            )
            overlap_rows.extend(
                {
                    "left_cohort": left,
                    "right_cohort": right,
                    "identity_policy": policy,
                    "canonical_identity": s,
                    "left_ligand_ids": a[s],
                    "right_ligand_ids": b[s],
                }
                for s in shared
            )
    for name in TARGETS:
        mat = residue_matrix(name)
        coverage.append(
            {
                "target": name,
                "matrix_rows": len(mat),
                "residue_features": mat.shape[1],
                "source_ids_missing_from_matrix": len(set(df.ligand_id) - set(mat.index)),
                "matrix_ids_absent_from_source": len(set(mat.index) - set(df.ligand_id)),
            }
        )
    output = result(
        "The public source supplies the 3171 labeled pesticide/AChE-inactive records and six residue "
        "matrices. Its 2749 unlabeled records cannot yet be matched to the final paper's 2465 docked metabolites. "
        "The separate reported flow for DS1 is 5211 raw → 5027 standardized → 2465 prepared; those counts "
        "are transcribed references, not recomputed curation or ligand-preparation outcomes.",
        [
            metric("source_records", len(df)),
            metric("labeled_records", int(df.activity.notna().sum())),
            metric("unlabeled_records", int(df.activity.isna().sum())),
        ],
        {
            "paper_cohort_flow": table(paper_flow, "Reported stages remain separate from computed source-cohort counts; no inclusion map is inferred", "paper_reported"),
            "paper_flow_arithmetic": table(arithmetic,
                "Arithmetic on reported aggregate counts, not reconstructed preparation: 8430−5636=2794 differs from section4.4's 2704 by90. No source records are changed."),
            "cohort_audit": table(
                pd.DataFrame(rows), "Exact isomeric identity; no silent salt or stereochemistry changes"
            ),
            "residue_matrix_coverage": table(
                pd.DataFrame(coverage), "ID-aligned coverage, including surplus matrix records"
            ),
            "cross_cohort_identity_overlap": table(
                pd.DataFrame(overlap_summary),
                "Exact and explicitly stereo-ignored sensitivities; counts of identities and records differ",
            ),
            "shared_identity_records": table(
                pd.DataFrame(overlap_rows),
                "All overlapping IDs retained; no deduplication or relabeling performed",
            ),
        },
        "cannabis.source_audit.v1",
        notes=[
            EvidenceNote(
                code="REPORTED_FLOW_AND_TRAINING_POPULATION",
                message="Reported standardized DS2+DS3 total 3403, while the bundled labeled total 3171 "
                "matches the reported docked reference counts 1680+1491. Equal counts do not prove identical "
                "molecules or identify the authors' exact graph-model training cohort and folds. Reported "
                "DS1 preparation coverage 2465/5027 is a count fraction, not evidence of chemical representativeness "
                "or independently confirmed natural occurrence. No missing inclusion decisions are reconstructed by truncation.",
            ),
            EvidenceNote(
                code="COHORT_OVERLAP_SCOPE",
                message="Overlapping structure identities "
                "cannot be treated as novel independent validation compounds. Overlap alone does not establish "
                "contamination or natural biosynthesis; occurrence metadata is needed. The stereo-ignored "
                "view is a declared sensitivity, not a replacement of stereochemically distinct molecules.",
            )
        ],
    )
    output.metadata.sources.extend(s for s in paper_sources.sources if s.name in {"paper.pdf", "paper_cohort_flow.csv"})
    output.artifacts.append(save_artifact(
        (DATA_ROOT / "cannabis_candidates/paper.pdf").read_bytes(), suffix="pdf", media_type="application/pdf",
        public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341"),
        display_name="cannabis_2026_source_paper.pdf",
    ))
    output.metadata.protocol_parameters["source_paper_artifact_role"] = "input reference document, not generated analysis"
    return output


def docking_comparisons():
    df = load()
    statistics = []
    tests = []
    for target, (column, _) in TARGETS.items():
        for label, cohort in [(1, "DS2_pesticides"), (0, "DS3_AChE_inactive")]:
            statistics.append(
                {"target": target, "cohort": cohort, **distribution(df.loc[df.activity == label, column])}
            )
        a = df.loc[df.activity == 1, column].to_numpy(float)
        b = df.loc[df.activity == 0, column].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Missing/nonfinite labeled docking scores; do not silently alter test cohorts")
        u = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        tests.append(
            {
                "target": target,
                "n_DS2": len(a),
                "n_DS3": len(b),
                "median_difference_DS2_minus_DS3": float(np.median(a) - np.median(b)),
                "mean_difference_DS2_minus_DS3": float(np.mean(a) - np.mean(b)),
                "u_statistic": float(u.statistic),
                "p_two_sided": float(u.pvalue),
            }
        )
    for row, q in zip(tests, bh_fdr([r["p_two_sided"] for r in tests])):
        row["p_fdr"] = float(q)
        row["significant_at_paper_0_001"] = bool(q < 0.001)
    return result(
        "Six DS2-versus-DS3 docking comparisons were recomputed with two-sided Mann–Whitney U tests "
        "and BH correction across six targets. Lower energies are docking features, not measured affinity. "
        "This comparison covers the two labeled cohorts only. The unavailable final DS1 map does not "
        "affect the completed DS2-versus-DS3 comparisons. No new docking is required for this score reanalysis. "
        "Both mean and median differences are returned: the published numerical range and OR28 value "
        "agree with the mean differences after rounding, although section4.4 defines delta as a median difference.",
        [metric("n_target_comparisons", len(tests), "DS2 versus DS3")],
        {
            "target_tests": table(pd.DataFrame(tests), "Distinct mean and median DS2-minus-DS3 differences, kcal/mol; Mann–Whitney tests distributions, not either summary alone"),
            "target_distributions": table(
                pd.DataFrame(statistics), "Full labeled source cohorts; docking units kcal/mol"
            ),
        },
        "cannabis.mann_whitney_six_targets.v2",
        notes=[EvidenceNote(code="PAPER_MEAN_MEDIAN_DISCREPANCY",
            message="Section4.4 explicitly defines delta=median(DS2)-median(DS3). On the deposited labeled "
                    "scores this gives OR28 +1.40 and a five-target range -0.70 to -1.20 kcal/mol. "
                    "The mean differences instead round to the published +1.20 and -0.70 to -1.01. "
                    "This numerical correspondence suggests a source definition/reporting mismatch; it does "
                    "not establish the authors' implementation. Neither statistic is relabeled as the other.")],
        status="success",
        analysis_scope="six labeled DS2-versus-DS3 source-score comparisons; not a new docking run or full-paper reproduction",
        unavailable_ds1_map_affects_this_comparison=False,
        alternative="two-sided",
        method="asymptotic",
        continuity_correction=True,
        multiplicity="Benjamini-Hochberg over six targets",
        paper_alpha=0.001,
    )


def select_residues(target="GLC1", split_index=0, inner_splits=5, inner_test_size=0.2, seed=42):
    train, test = split(split_index)
    matrix = residue_matrix(target)
    if not set(train.ligand_id).union(test.ligand_id).issubset(matrix.index):
        raise ValueError("Missing entire ligand rows cannot be silently interpreted as zero interaction")
    x = matrix.loc[train.ligand_id].to_numpy(float)
    fitted = fit_residue_selector(
        x,
        train.activity.to_numpy(int),
        list(matrix),
        inner_splits=inner_splits,
        inner_test_size=inner_test_size,
        seed=seed,
    )
    heldout = fitted.transform(matrix.loc[test.ligand_id].to_numpy(float), list(matrix))
    names = [fitted.feature_names[i] for i in fitted.selected_indices]
    heldout_frame = pd.DataFrame(heldout, columns=names)
    heldout_frame.insert(0, "ligand_id", test.ligand_id.to_numpy())
    train_frame = pd.DataFrame(fitted.transform(x, list(matrix)), columns=names)
    train_frame.insert(0, "ligand_id", train.ligand_id.to_numpy())
    registry = pd.concat(
        [
            train[["ligand_id", "activity"]].assign(partition="train"),
            test[["ligand_id", "activity"]].assign(partition="test"),
        ],
        ignore_index=True,
    )
    identity_sets = [{canonical_smiles(s) for s in frame.SMILES} for frame in [train, test]]
    scaffold_sets = [{scaffold(s) for s in frame.SMILES} for frame in [train, test]]
    split_hash = hashlib.sha256(registry.to_csv(index=False).encode()).hexdigest()
    r = result(
        f"{target}, source split {split_index}: the Appendix A selector retained "
        f"{len(names)} original residue-energy columns after the 98% contribution gate. "
        "All selection used outer training rows; held-out energies were only sliced by the fitted selection.",
        [
            metric(k, v, f"{target}, split_{split_index:02d} outer training")
            for k, v in fitted.diagnostics.items()
        ]
        + [
            metric("n_test", len(test)),
            metric("train_test_ligand_id_overlap", len(set(train.ligand_id) & set(test.ligand_id))),
            metric("train_test_canonical_identity_overlap", len(identity_sets[0] & identity_sets[1])),
            metric("train_test_murcko_overlap", len(scaffold_sets[0] & scaffold_sets[1])),
        ],
        {
            "selected_residues": table(
                fitted.ranking.set_index("residue").loc[
                    names, ["rmt_rank", "energy_rank_within_top_m", "energetic_contribution"]
                ].reset_index(),
                "Complete selected residue labels in the exact column order of exported train/test energies; "
                "this is the selected subset, not a preview of all ranked residues",
            ),
            "ranked_residues": table(
                fitted.ranking, "Original target and chain-specific residue labels retained"
            ),
            "inner_m_scan": table(
                fitted.scan, "One-dimensional cumulative raw energy; LR trained only on inner training fold"
            ),
            "eigenspectrum": table(
                fitted.spectrum, "Training correlation spectrum and Marchenko–Pastur edge"
            ),
        },
        "cannabis.rmt_appendix_a.v1",
        notes=[
            EvidenceNote(
                code="UNREPORTED_NUMERICAL_SETTINGS",
                message="The final paper does not identify the candidate m grid, inner split count/fraction, "
                "seed, tie rule or logistic-regression regularization. This implementation scans all m, uses "
                "the explicit inner settings in metadata, LR C=1, and smaller-m exact ties. Source outer splits "
                "come from the public historical bundle; their identity with final-paper folds is unverified. "
                "This is a transparent implementation of the published method, not an exact numerical replay "
                "of an unavailable final training run.",
            )
        ],
        target=target,
        split_index=split_index,
        split_sha256=split_hash,
        inner_splits=inner_splits,
        inner_test_size=inner_test_size,
        seed=seed,
        candidate_m="all integers 1..p",
        exact_tie_rule="smaller m",
        logistic_C=1.0,
        standardization="training mean and sample standard deviation (ddof=1)",
        rank_fit_scope="outer training rows only",
        ranking="RMT score only",
        selection_metric="inner validation ROC-AUC",
        contribution_fraction=0.98,
        final_features="selected original residue-energy columns",
        test_used_in_selection=False,
    )
    payload = {
        "metadata": r.metadata.model_dump(mode="json"),
        "split_registry": json.loads(registry.to_json(orient="records")),
        "train_selected_energies": json.loads(train_frame.to_json(orient="records", double_precision=15)),
        "test_selected_energies": json.loads(heldout_frame.to_json(orient="records", double_precision=15)),
    }
    r.artifacts.append(
        save_json(
            payload,
            public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341"),
            display_name=f"cannabis_{target}_split{split_index:02d}_selected_energies.json",
        )
    )
    return r
