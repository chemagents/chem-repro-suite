"""Article-specific population selection over shared chemistry/statistics."""

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common.chemistry import morgan_fingerprints, scaffold
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import provenance, table
from ..common.statistics import distribution, permutation_separability, profile_pca
from ..common.validation import regression_oof
from .data import CLASSES, ENDPOINTS, SATURATED, limitations, load, source_dataset


def result(finding, metrics, tables, *, protocol, notes=(), **params):
    return AnalysisResult(
        status="success",
        answer=Answer(
            finding=finding, metrics=metrics, tables=tables, limitations=limitations() + list(notes)
        ),
        metadata=provenance("heracleum", protocol, **params),
    )


def metric(name, value, population="deposited master snapshot", unit=None):
    return Metric(name=name, value=value, population=population, unit=unit)


def dataset_audit():
    df, targets = load()
    df["master_source_murcko_matches"] = df.canonical_smiles.map(scaffold).eq(
        df.SMILES_standardized.map(scaffold)
    )
    bad = df.loc[
        ~df.identity_verified | ~df.generic_scaffold_matches,
        [
            "synID",
            "dataset_membership",
            "canonical_smiles",
            "SMILES_standardized",
            "reported_name",
            "identity_verified",
            "generic_scaffold_matches",
            "master_source_murcko_matches",
        ],
    ]
    metrics = {
        "ds1": int(df.ds1_member.sum()),
        "ds2": int(df.ds2_member.sum()),
        "union": len(df),
        "overlap": int((df.ds1_member & df.ds2_member).sum()),
        "valid_docking": int(df.has_valid_44_docking_scores.sum()),
        "docking_targets": len(targets),
        "structure_identity_conflicts": int((~df.identity_verified).sum()),
        "generic_scaffold_conflicts": int((~df.generic_scaffold_matches).sum()),
        "master_source_murcko_conflicts": int((~df.master_source_murcko_matches).sum()),
        "unusable_archived_docking_profiles": int((~df.has_valid_44_docking_scores).sum()),
        "parsable_master_smiles": len(df),
    }
    omitted = df.loc[~df.has_valid_44_docking_scores,
                     ["synID", "dataset_membership", "canonical_smiles"]].copy()
    omitted["all_44_archived_scores_zero"] = (df.loc[omitted.index, targets] == 0).all(axis=1)
    return result(
        f"The master contains {len(df)} parsable molecular records, "
        f"{metrics['structure_identity_conflicts']} master-versus-source canonical SMILES disagreements at the same synID. "
        f"{metrics['master_source_murcko_conflicts']} master/source pairs differ even in stereo-free Murcko scaffolds. "
        f"Separately, {metrics['generic_scaffold_conflicts']} stored generic-scaffold labels disagree with "
        "scaffolds recomputed from the master structure. These are two different comparisons. "
        f"{metrics['valid_docking']} records have source-marked usable archived docking profiles; "
        f"the other {len(omitted)} profiles contain all-zero scores. This does not establish geometry or valence errors.",
        [metric(k, v, population=("source-defined usable archived 44-target score profiles; not a new 3D check"
                                 if k == "valid_docking" else "deposited master snapshot"))
         for k, v in metrics.items()],
        {
            "identity_audit": table(
                bad, "Both structures retained; master_source_murcko_matches compares master with DS1/DS2, "
                "whereas generic_scaffold_matches compares a stored label with the master only. "
                "reported_name is unverified for conflicts"
            ),
            "unusable_archived_profiles": table(omitted, "All-zero archived scores, not invalid molecular graphs"),
        },
        protocol="heracleum.identity_audit.v3",
        notes=[EvidenceNote(code="ARCHIVED_PROFILE_NOT_GEOMETRY_VALIDATION", severity="info",
                            message="All master and source SMILES parse successfully for the identity audit. "
                            "The 18 unusable archived docking profiles have all-zero scores; this is not evidence "
                            "of invalid valence, bonds or 3D geometry. No new ligand preparation or docking was run."),
               EvidenceNote(code="IDENTITY_COMPARISON_DEFINITION", severity="info",
                            message="Identity conflicts compare canonical isomeric SMILES in the master with "
                            "canonical isomeric SMILES in DS1/DS2 for the same synID. They are not SMILES-versus-" 
                            "InChI disagreements, and this audit does not establish which structure generated each prediction."),
               EvidenceNote(code="DISTINCT_SCAFFOLD_COMPARISONS", severity="info",
                            message="master_source_murcko_conflicts counts differences between stereo-free, "
                            "element/bond-preserving Murcko scaffolds of master and source structures. Such "
                            "differences cannot be explained by stereochemistry alone. generic_scaffold_conflicts "
                            "instead counts stored generic labels that disagree with recomputation from the "
                            "master; it does not count how many master/source structures have different cores.")],
        identity="master versus source isomeric canonical SMILES matched by unique synID",
    )


def chemical_space():
    metrics = []
    rows = []
    summary = []
    for subset in ["DS1", "DS2"]:
        df = source_dataset(subset)
        cores = [scaffold(s, retain_stereo=True) for s in df.SMILES_standardized]
        generics = [scaffold(s, generic=True, retain_stereo=True) for s in df.SMILES_standardized]
        counts = Counter(cores)
        ranked = counts.most_common()
        n = len(df)
        n50 = int(np.searchsorted(np.cumsum([v for _, v in ranked]), n / 2) + 1)
        rec = {
            "subset": subset,
            "n": n,
            "murcko_scaffolds": len(counts),
            "generic_scaffolds": len(set(generics)),
            "singleton_scaffolds_total": sum(v == 1 for v in counts.values()),
            "singleton_scaffolds_outside_top15": sum(v == 1 for _, v in ranked[15:]),
            "scaffolds_for_50_percent": n50,
            "top2_n": sum(v for _, v in ranked[:2]),
            "top5_n": sum(v for _, v in ranked[:5]),
            "top15_n": sum(v for _, v in ranked[:15]),
        }
        summary.append(rec)
        for k in [2, 5, 15]:
            metrics.append(metric(f"{subset}_top{k}_fraction", rec[f"top{k}_n"] / n, population=subset))
        for rank, (core, count) in enumerate(ranked[:15], 1):
            rows.append(
                {
                    "subset": subset,
                    "rank": rank,
                    "scaffold_smiles": core,
                    "n_molecules": count,
                    "fraction": count / n,
                }
            )
        if subset == "DS1":
            psoralen = scaffold("O=c1ccc2cc3ccoc3cc2o1", retain_stereo=True)
            angelicin = scaffold("O=c1ccc2ccc3occc3c2o1", retain_stereo=True)
            metrics.extend(
                [
                    metric("DS1_psoralen_scaffold_n", counts[psoralen], population=subset),
                    metric("DS1_angelicin_scaffold_n", counts[angelicin], population=subset),
                    metric(
                        "DS1_psoralen_angelicin_scaffold_ratio",
                        counts[psoralen] / counts[angelicin],
                        population=subset,
                    ),
                ]
            )
    return result(
        "Scaffold concentration was recomputed from the original DS1/DS2 structure files. "
        "The psoralen/angelicin ratio describes exact Murcko frameworks, not all substructure matches.",
        metrics,
        {
            "diversity": table(pd.DataFrame(summary), "Scaffold counts and cumulative coverage"),
            "top_scaffolds": table(pd.DataFrame(rows), "Top 15 frameworks in each original dataset"),
        },
        protocol="heracleum.scaffold_space.v1",
        source="DS1_2008.csv and DS2_102.csv",
        murcko_stereochemistry="RDKit default retained, as in the original notebook",
    )


@lru_cache(maxsize=1)
def docking_components():
    df, targets = load()
    valid = df.loc[df.has_valid_44_docking_scores].copy()
    matrix = np.minimum(valid[targets].to_numpy(float), 0)
    pca = profile_pca(matrix)
    rho = float(spearmanr(pca["scores"][:, 0], matrix.mean(axis=1)).statistic)
    if rho < 0:
        pca["scores"][:, 0] *= -1
        pca["loadings"][:, 0] *= -1
    for k in range(3):
        valid[f"PC{k + 1}"] = pca["scores"][:, k]
    return valid, targets, matrix, pca, abs(rho)


def docking_profile():
    valid, targets, matrix, pca, rho = docking_components()
    components = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca["explained_variance_ratio"]))],
            "variance_fraction": pca["explained_variance_ratio"],
            "cumulative_fraction": np.cumsum(pca["explained_variance_ratio"]),
        }
    )
    loadings = pd.DataFrame(pca["loadings"][:, :3], columns=["PC1", "PC2", "PC3"])
    loadings.insert(0, "target", [t[5:] for t in targets])
    strong = (matrix < -10).sum(axis=1)
    associations = [
        {
            "endpoint": name,
            "n_records": len(valid),
            "spearman_mean_docking": float(spearmanr(matrix.mean(axis=1), valid[col]).statistic),
        }
        for name, col in ENDPOINTS.items()
    ]
    return result(
        f"PC1 explains {pca['explained_variance_ratio'][0]:.1%} of docking variance; "
        f"{pca['components_to_threshold']} components reach 80%. Mean docking tracks PC1 with rho={rho:.3f}.",
        [
            metric("n_molecules", len(valid)),
            metric("n_targets", len(targets)),
            metric("components_for_80_percent", pca["components_to_threshold"]),
            metric("pc1_mean_docking_spearman", rho),
            metric("median_n_strong", float(np.median(strong))),
            metric("mean_n_strong", float(strong.mean())),
        ],
        {
            "pca_variance": table(components, "Column-z-scaled docking PCA"),
            "target_loadings": table(loadings, "All target loadings; PC1 oriented with mean docking score"),
            "docking_toxicity_associations": table(
                pd.DataFrame(associations),
                "Descriptive Spearman coefficients on the 2055 valid docking profiles; not independent biological validation",
            ),
        },
        protocol="heracleum.docking_pca.v1",
        notes=[EvidenceNote(
            code="PCA_INTERPRETATION_BOUNDARY", severity="info",
            message="The variance remaining after PC1 is the residual of the best one-dimensional linear PCA "
                    "reconstruction, not an exact measured information loss caused by using the arithmetic mean. "
                    "A Spearman correlation below one does not make the mean identical to PC1. Four components "
                    "are needed for 80% variance here. Loading signs describe covariance directions under the "
                    "chosen axis orientation; an opposite sign is not orthogonality or experimental selectivity. "
                    "Docking scores and ADMETlab endpoints are calculated outputs, not measured binding or toxicity.",
        )],
        identity_policy="published_snapshot",
        positive_scores_capped_at=0,
        standardize=True,
    )


def toxicity_landscape():
    df, _ = load()
    x = df[list(ENDPOINTS.values())].copy()
    x.columns = list(ENDPOINTS)
    pca = profile_pca(x)
    rows = [
        {
            "endpoint": e,
            **distribution(x[e]),
            "iqr": float(x[e].quantile(0.75) - x[e].quantile(0.25)),
            "fraction_gt_0_9": float((x[e] > 0.9).mean()),
            "excluded_from_tiering": e in SATURATED,
        }
        for e in ENDPOINTS
    ]
    cor = x.corr(method="spearman").rename_axis("endpoint").reset_index()
    cumulative = np.cumsum(pca["explained_variance_ratio"])
    needed = pca["components_to_threshold"]
    variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ENDPOINTS))],
            "variance_fraction": pca["explained_variance_ratio"],
            "cumulative_fraction": cumulative,
        }
    )
    statistics = pd.DataFrame(rows)
    statistics["iqr_rank_ascending"] = statistics["iqr"].rank(method="min").astype(int)
    return result(
        f"The minimum number of components needed to reach at least 80% of predicted toxicity variance is {needed}: "
        f"the first {needed - 1} explain {cumulative[needed - 2] if needed > 1 else 0:.4%}, "
        f"and the first {needed} explain {cumulative[needed - 1]:.4%}. "
        "80% is a threshold, not a claim that the cumulative variance equals exactly 80%. "
        "This poor compression does not establish independence of endpoints. "
        "IQR ranks describe dispersion only, not accuracy or a new endpoint-exclusion rule.",
        [metric("n_molecules", len(x)), metric("components_for_80_percent", pca["components_to_threshold"])],
        {
            "endpoint_statistics": table(
                statistics, "All ten endpoints retained; IQR rank1 is narrowest, not an accuracy ranking"
            ),
            "spearman_correlations": table(cor, "Pairwise endpoint Spearman correlations"),
            "pca_variance": table(variance, "PCA of column-standardized model probabilities"),
        },
        protocol="heracleum.toxicity_landscape.v1",
        identity_policy="published_snapshot",
    )


def prioritize(identity_policy="strict", strong_threshold=-10.0):
    from .alerts import alert_audit

    if identity_policy not in {"strict", "published_snapshot"}:
        raise ValueError("Invalid identity policy")
    if not np.isfinite(strong_threshold) or not -30 <= strong_threshold < 0:
        raise ValueError("Invalid docking threshold")
    all_rows, targets = load()
    df = all_rows.loc[all_rows.ds2_member]
    informative = {k: v for k, v in ENDPOINTS.items() if k not in SATURATED}
    medians = df[list(informative.values())].median()
    herg_threshold = float(np.nanmedian(all_rows[targets].to_numpy(float)))
    rows = []
    for r in df.itertuples():
        probs = {e: float(getattr(r, c)) for e, c in informative.items()}
        favorable = [e for e, c in informative.items() if probs[e] < medians[c]]
        scores = np.array([getattr(r, t) for t in targets], float)
        valid = bool(r.has_valid_44_docking_scores) and np.isfinite(scores).all()
        n_strong = int((scores < strong_threshold).sum()) if valid else None
        complete = valid and pd.notna(r.admetai_pains_alert) and pd.notna(r.admetai_brenk_alert)
        usable = complete and (r.identity_verified or identity_policy == "published_snapshot")
        tier = None
        if usable:
            tier = (
                1
                if len(favorable) >= 6
                and n_strong <= 2
                and r.admetai_brenk_alert <= 1
                and r.admetai_pains_alert == 0
                else (2 if len(favorable) >= 5 and r.admetai_pains_alert == 0 else 3)
            )
        rows.append(
            {
                "synID": int(r.synID),
                "name": r.trivial_name if pd.notna(r.trivial_name) else None,
                "identity_verified": bool(r.identity_verified),
                "tier": tier,
                "n_favorable": len(favorable),
                "favorable_endpoints": favorable,
                "n_strong": n_strong,
                "pains": r.admetai_pains_alert,
                "brenk": r.admetai_brenk_alert,
                "probabilities": probs,
                "herg_docking_score": float(r.dock_KCNH2),
                "herg_weak_source_rule": bool(r.dock_KCNH2 > herg_threshold),
                "herg_profile_and_identity_verified": bool(valid and r.identity_verified),
            }
        )
    counts = Counter(str(r["tier"]) if r["tier"] else "unranked" for r in rows)
    alerts = alert_audit()
    rows.sort(key=lambda r: (r["tier"] or 4, -r["n_favorable"], r["synID"]))
    output = result(
        f"DS2 tiers in {identity_policy} mode: {dict(counts)}. Tiers are relative experimental "
        "follow-up priorities, not safety classes. "
        f"Recomputed alert counts match the archive for {int(alerts.original_DS2_counts_match_archive.sum())}/102 "
        f"original DS2 structures and {int(alerts.master_counts_match_archive.sum())}/102 master structures.",
        [metric("tier_" + k, int(v), population="DS2") for k, v in counts.items()],
        {
            "candidates": table(pd.DataFrame(rows, dtype=object), "Every criterion and identity flag retained"),
            "median_thresholds": table(
                pd.DataFrame([{"endpoint": e, "median": medians[c]} for e, c in informative.items()]),
                "All 102 deposited DS2 records define thresholds in both modes",
            ),
            "structural_alert_recomputation": table(
                alerts,
                "RDKit catalog-entry counts/names for master prediction structures and original DS2 separately; archived flags still define reproduced tiers",
            ),
        },
        protocol="heracleum.candidate_policy.v1",
        identity_policy=identity_policy,
        strong_threshold=strong_threshold,
        herg_panel_median_kcal_mol=herg_threshold,
        notes=[
            EvidenceNote(
                code="ALERT_RECOMPUTATION_SCOPE",
                message="PAINS and Brenk were recomputed "
                "using the pinned RDKit catalogs. Counts refer to matching catalog entries, not the "
                "number of fragment occurrences. The source's archived counts still define the reproduced "
                "tiers; alerts on original DS2 structures are not transplanted onto predictions for "
                "conflicting master structures. Alert absence is not experimental safety evidence.",
            ),
            EvidenceNote(
                code="MEDIAN_REFERENCE_CONFLICTS",
                message="Even strict-mode medians retain "
                "the 102-record deposited reference, so they require reassessment after identity correction.",
            ),
        ],
    )
    from .priority_reference import attach_priority_reference

    return attach_priority_reference(output)


def separability(permutations=10000):
    df, *_ = docking_components()
    df = df.loc[df.scaffold_class.isin(CLASSES)]
    axes = {name: df[col].to_numpy(float) for name, col in ENDPOINTS.items()}
    axes.update({name: df[name].to_numpy(float) for name in ["PC1", "PC2", "PC3"]})
    rows = permutation_separability(
        axes, df.scaffold_class.to_numpy(), CLASSES, permutations=permutations, seed=42
    )
    n_sig = sum(r["p_fdr"] < 0.01 for r in rows)
    return result(
        f"{n_sig}/13 axes exceed random scaffold separation at FDR p<0.01. This tests "
        "structural association, not biological validity or transferability.",
        [
            metric("n_records", len(df)),
            metric("n_axes", len(rows)),
            metric("n_significant_fdr_0_01", n_sig),
            metric(
                "mean_toxicity_ks", float(np.mean([r["observed_ks"] for r in rows if r["axis"] in ENDPOINTS]))
            ),
            metric("mean_null_ks", float(np.mean([r["null_mean_ks"] for r in rows]))),
        ],
        {
            "axis_tests": table(pd.DataFrame(rows), "All 13 axes, shared cohort and BH family"),
            "groups": table(
                df.scaffold_class.value_counts().rename_axis("class").reset_index(name="n"),
                "Other excluded; five scaffold groups retained",
            ),
        },
        protocol="heracleum.separability.v1",
        permutations=permutations,
        seed=42,
        p_value="(exceedances+1)/(permutations+1)",
        identity_policy="published_snapshot",
    )


@lru_cache(maxsize=8)
def _transfer(endpoint, model, identity_policy, bootstrap):
    if endpoint not in {"neurotoxicity", "eye_corrosion"}:
        raise ValueError("Unsupported paper endpoint")
    df, _ = load(identity_policy)
    fps = morgan_fingerprints(df.canonical_smiles.tolist(), chirality=False)
    x = np.asarray([np.asarray(fp) for fp in fps], float)
    groups = (
        df.canonical_smiles.map(lambda s: scaffold(s, generic=True, retain_stereo=True)).to_numpy()
        if identity_policy == "strict"
        else df.generic_murcko_scaffold.fillna("none").to_numpy()
    )
    from .cv_registry import reference_fold_assignments
    ids = df.synID.astype(str).tolist()
    assignments = reference_fold_assignments(ids, groups, identity_policy)
    fitted = regression_oof(
        x,
        df[ENDPOINTS[endpoint]].to_numpy(float),
        groups,
        df.synID.astype(str).tolist(),
        model=model,
        bootstrap=bootstrap,
        seed=42,
        fold_assignments=assignments,
    )
    return fitted, len(df)


def transfer(
    endpoint="neurotoxicity",
    model="RF",
    identity_policy="strict",
    bootstrap=1000,
    include_oof=False,
):
    cache_before = _transfer.cache_info().hits
    fitted, n = _transfer(endpoint, model, identity_policy, bootstrap)
    rows = [
        {"scheme": scheme, **{k: v for k, v in r.items() if k not in {"fold_audit", "oof"}}}
        for scheme, r in fitted.items()
    ]
    tables = {
        "validation_metrics": table(
            pd.DataFrame(rows), "Pooled OOF metrics; fixed-prediction molecule bootstrap"
        ),
        "fold_audit": table(
            pd.DataFrame([row for r in fitted.values() for row in r["fold_audit"]]),
            "Generic scaffold overlap checked for every fold",
        ),
    }
    if include_oof:
        tables["oof_predictions"] = table(
            pd.DataFrame([{"scheme": s, **r} for s, v in fitted.items() for r in v["oof"]]),
            "Full per-molecule out-of-fold predictions",
        )
    output = result(
        f"{model} {endpoint}: random R²={fitted['random']['r2']:.3f}, scaffold R²="
        f"{fitted['scaffold']['r2']:.3f}. The target is ADMETlab output, not experimental toxicity.",
        [metric("n_records", n), metric("delta_r2", fitted["random"]["r2"] - fitted["scaffold"]["r2"])],
        tables,
        protocol="heracleum.surrogate_cv.v2",
        model=model,
        endpoint=endpoint,
        target_column=ENDPOINTS[endpoint],
        fold_assignment="frozen_reference_implementation_registry_v1_not_author_supplied",
        identity_policy=identity_policy,
        bootstrap=bootstrap,
        seed=42,
        morgan_radius=2,
        morgan_bits=2048,
        notes=[
            EvidenceNote(
                code="REFERENCE_FOLD_REGISTRY", severity="info",
                message="Fold assignments are frozen from the reference implementation to avoid "
                "platform-dependent GroupKFold tie ordering. These are not author-supplied folds. "
                "Input identities and group membership are checked; models are fitted anew.",
            ),
            EvidenceNote(
                code="BOOTSTRAP_SCOPE",
                message="Molecule bootstrap of fixed OOF predictions; "
                "no model refitting, no within-scaffold dependence adjustment, unknown overlap with ADMETlab training.",
            ),
            EvidenceNote(
                code="EYE_ENDPOINT_DISTINCTION",
                severity="info",
                message="eye_corrosion (коррозия глаз; source admetlab_tox_ec) and "
                "eye_irritation (раздражение глаз; source admetlab_tox_ei) are separate ADMETlab "
                "prediction columns. This transfer protocol uses eye_corrosion, not eye_irritation. "
                "Do not translate or interpret one as the other.",
            ),
            EvidenceNote(
                code="TRANSFER_COMPARISON_SCOPE",
                message="delta_r2 is random R² minus scaffold R² for the same model and endpoint. "
                "A smaller drop from a lower random-CV baseline alone does not establish greater "
                "robustness of one model: compare absolute scaffold-CV performance as well. "
                "These estimates quantify performance under the declared splits; no reliability "
                "threshold or causal explanation of differences between models was tested.",
            ),
        ],
    )
    output.metadata.reused_from_cache = _transfer.cache_info().hits > cache_before
    return output
