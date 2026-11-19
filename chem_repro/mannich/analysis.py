"""Recompute the Mannich paper's analyses from manually curated source records."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import unquote

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import skew

from ..common.chemistry import canonical_smiles, catalyst_identity, ee_to_ddg, scaffold
from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from ..common.statistics import distribution

ROLE_COLUMNS = ("electrophile", "nucleophile", "organocatalyst", "product")


def normalize_doi(value: str) -> str:
    text = unquote(str(value)).strip()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.IGNORECASE).lower()


@lru_cache(maxsize=1)
def _load():
    verified_manifest("mannich")
    df = pd.read_csv(DATA_ROOT / "mannich/mannich_dataset.csv", keep_default_na=False, na_values=[""])
    required = {
        *ROLE_COLUMNS,
        "paper_doi",
        "axial_configuration",
        "ee",
        "ddG_kcal_mol",
        "temperature",
        "yield",
        "time",
        "organocatalyst_class",
        "organocatalyst_3d_sdf",
        "solvent",
    }
    if required - set(df):
        raise ValueError(f"Missing source columns: {sorted(required - set(df))}")
    df.insert(0, "reaction_id", [f"mannich:{i + 1:04d}" for i in range(len(df))])
    df["normalized_doi"] = df.paper_doi.map(normalize_doi)
    for role in ROLE_COLUMNS:
        # Refuse an invalid structure rather than silently reducing the article's cohort.
        df[role + "_canonical"] = df[role].map(canonical_smiles)
    df["catalyst_identity"] = [
        catalyst_identity(s, a) for s, a in zip(df.organocatalyst, df.axial_configuration)
    ]
    return df


def load():
    return _load().copy()


def _result(finding, metrics, tables, *, protocol, notes=(), status="success", **params):
    return AnalysisResult(
        status=status,
        answer=Answer(finding=finding, metrics=metrics, tables=tables, limitations=list(notes)),
        metadata=provenance("mannich", protocol, **params),
    )


def _metric(name, value, *, unit=None, population="all curated reaction records"):
    return Metric(name=name, value=value, unit=unit, population=population)


def overview(include_audit=True):
    df = load()
    counts = {
        "reaction_records": len(df),
        "source_fields": len(pd.read_csv(DATA_ROOT / "mannich/mannich_dataset.csv", nrows=0).columns),
        "source_publications": df.normalized_doi.nunique(),
        "raw_doi_strings": df.paper_doi.nunique(),
        "unique_catalysts_including_axial": df.catalyst_identity.nunique(),
        "unique_catalyst_isomeric_smiles": df.organocatalyst_canonical.nunique(),
        "unique_nucleophiles": df.nucleophile_canonical.nunique(),
        "unique_electrophiles": df.electrophile_canonical.nunique(),
    }
    pubs = df.groupby("normalized_doi", sort=True).agg(n_reactions=("reaction_id", "size")).reset_index()
    result = _result(
        f"{len(df)} manually curated reaction records from {counts['source_publications']} "
        f"normalized DOI identities; {counts['unique_catalysts_including_axial']} catalysts "
        "when axial configuration is part of identity.",
        [_metric(k, int(v)) for k, v in counts.items()],
        {"publications": table(pubs, "Reaction counts by normalized DOI identity")},
        protocol="mannich.dataset_identity_and_quality.v3",
        notes=[
            EvidenceNote(
                code="MANUAL_SOURCE_REVIEW",
                severity="info",
                message="Reaction-image/OCR extraction is manual. This tool validates deposited records; "
                "it does not claim to have re-read all 153 publications.",
            )
        ],
        catalyst_identity="canonical isomeric SMILES + explicit R/S/none axial flag",
        doi_normalization="strip DOI resolver prefix, URL-decode, case-fold",
    )
    result.answer.finding += f" Declared dataset-article DOI: {result.metadata.paper_doi}."
    if include_audit:
        from ..common.artifacts import save_table
        from .quality import audit_records

        tables, full = audit_records(df)
        result.answer.tables.update(tables)
        groups = tables["repeated_input_groups"].rows
        grouped_records = sum(row["n_records"] for row in groups)
        result.answer.metrics.extend([
            _metric("repeated_input_groups_with_different_outcomes", len(groups), unit="groups",
                    population="groups of identical recorded inputs and conditions with different outcomes"),
            _metric("records_in_repeated_input_groups", grouped_records, unit="records",
                    population="reaction records belonging to the repeated-input groups"),
        ])
        result.answer.finding += (
            " The automated record audit retains all rows and exposes diagnostics for source review. "
            f"There are {len(groups)} repeated-input groups containing {grouped_records} records; "
            "record counts and group counts are distinct quantities."
        )
        result.artifacts.append(
            save_table(
                table(full, "All4015 source records and diagnostic flags"),
                port=7343,
                display_name="mannich_full_record_audit.csv",
            )
        )
        result.answer.limitations.append(
            EvidenceNote(
                code="RECORD_AUDIT_SCOPE",
                message="Atom inventories compare one recorded electrophile and one nucleophile against the product, "
                "not a balanced reaction equation. Byproducts, counterions, decarboxylation, protecting-group "
                "changes and in-situ imine formation can explain differences. Unassigned carbon stereo and "
                "differing outcomes for repeated recorded inputs require source review; neither is silently "
                "corrected or excluded. Additive names/formulas are not assumed to be SMILES. RDKit charge "
                "parsing does not validate solution protonation states or the original reaction drawing.",
            )
        )
        result.metadata.protocol_parameters["record_audit"] = (
            "preserve all rows; diagnostics only; no inferred atom mapping"
        )
    return result


def target_distribution():
    df = load()
    calculated = np.array([ee_to_ddg(float(e), float(t)) for e, t in zip(df.ee, df.temperature)])
    delta = calculated - df.ddG_kcal_mol.to_numpy(float)
    summary = pd.DataFrame(
        [
            {"target": name, **distribution(values)}
            for name, values in [
                ("ee_percent", df.ee),
                ("stored_abs_ddg_kcal_mol", df.ddG_kcal_mol),
                ("recomputed_abs_ddg_kcal_mol", calculated),
            ]
        ]
    )
    # Retain the population moment already exposed by the tool and name the
    # sample-adjusted estimator separately; pandas and scipy defaults differ.
    summary["skewness_bias_corrected"] = [
        float(skew(np.asarray(values, float), bias=False))
        for values in (df.ee, df.ddG_kcal_mol, calculated)
    ]
    mismatches = df.loc[np.abs(delta) > 1e-8, ["reaction_id", "ee", "temperature", "ddG_kcal_mol"]].copy()
    mismatches["recomputed_ddg"] = calculated[np.abs(delta) > 1e-8]
    records = df[["reaction_id", "ee", "temperature", "ddG_kcal_mol"]].copy()
    records["recomputed_ddg"] = calculated
    records["absolute_residual_kcal_mol"] = np.abs(delta)
    nonzero = df.ddG_kcal_mol != 0
    relative = np.full(len(df), np.nan)
    relative[nonzero] = np.abs(delta[nonzero] / df.ddG_kcal_mol[nonzero]) * 100
    records["absolute_relative_error_percent"] = relative
    return _result(
        "The Eyring transformation is recomputed from each record's ee and absolute temperature. "
        "It gives the magnitude of enantiomeric discrimination; no product-configuration sign is inferred.",
        [
            _metric("ddg_max_absolute_residual", float(np.max(np.abs(delta))), unit="kcal/mol"),
            _metric("records_at_99_9_percent_cap", int((df.ee == 99.9).sum())),
            _metric("ddg_mismatches_above_1e_8", len(mismatches)),
            _metric("recomputed_records", len(records)),
            _metric("nonzero_ddg_records_for_relative_error", int(nonzero.sum())),
            _metric("zero_ddg_records", int((~nonzero).sum())),
            _metric("median_absolute_relative_error_percent", float(np.median(relative[nonzero])), unit="%", population="curated reaction records with nonzero stored absolute ddG"),
        ],
        {
            "target_statistics": table(summary, "skewness: scipy bias=True; skewness_bias_corrected: bias=False; neither is silently substituted"),
            "conversion_records": table(records, "Every retained reaction and its independently recomputed |ddG|; relative error is null for a zero reference"),
            "conversion_disagreements": table(mismatches, "ONLY mismatches above 1e-8 kcal/mol; zero rows means no disagreements, not missing computation"),
        },
        protocol="mannich.eyring_magnitude.v2",
        gas_constant_kcal_mol_K=0.001987,
        kelvin_offset=273.15,
        ee_cap_percent=99.9,
        notes=[
            EvidenceNote(code="COMPLETE_ROW_COMPARISON", severity="info",
                message="All retained records were recomputed. conversion_records is the full per-reaction table; "
                "conversion_disagreements is only its mismatch subset. An empty mismatch table indicates agreement. "
                "Relative error is undefined for a zero reference and is left null; its median uses nonzero references only."),
            EvidenceNote(code="EE_VERSUS_DR", severity="info",
                message="ee compares enantiomer proportions; dr compares diastereomer proportions. "
                "Diastereomers are not mirror images and can differ at only one of multiple stereogenic centers "
                "(epimers); they do not have to differ at more than one center. "
                "Definitions: https://doi.org/10.1351/goldbook.D01679 and https://doi.org/10.1351/goldbook.E02167."),
            EvidenceNote(
                code="TARGET_SCOPE",
                message="dr is stored separately and is not used "
                "in |ddG|. Values at the cap do not reveal which original observations were exactly 100% ee.",
            ),
            EvidenceNote(
                code="PUBLICATION_BIAS",
                message="Reduced target skew does not establish "
                "improved model generalization; the source favors high-selectivity published reactions.",
            ),
        ],
    )


def conditions():
    df = load()
    stats = pd.DataFrame(
        [
            {"field": name, **distribution(pd.to_numeric(df[name], errors="raise"))}
            for name in ["temperature", "yield", "time", "organocatalyst_concentration(mol%)"]
        ]
    )

    def norm_solvent(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        if re.search(r"[/+]|\sand\s|\d+\s*[:/]\s*\d+", s, re.IGNORECASE):
            return "mixtures"
        if re.search("xylene", s, re.IGNORECASE):
            return "xylenes"
        return s.lower()

    categories = df.solvent.map(norm_solvent)
    counts = categories.value_counts().rename_axis("solvent_category").reset_index(name="n_reactions")
    raw = df.solvent.value_counts().rename_axis("source_solvent_text").reset_index(name="n_reactions")
    labels = df.solvent.str.strip().str.lower()
    simple_labels = labels[~labels.str.contains(r"[/+]|\sand\s|\d+\s*[:/]\s*\d+", regex=True)]
    solvent_labels_excluding_neat = int(simple_labels.loc[simple_labels != "neat"].nunique())
    return _result(
        "Condition distributions preserve reaction-level records and the notebook's solvent "
        "category normalization. Mixtures are not counted as a single pure solvent identity.",
        [
            _metric("distinct_source_solvent_strings", int(df.solvent.nunique())),
            _metric("normalized_solvent_categories", int(categories.nunique())),
            _metric("nonmixture_solvent_labels_excluding_neat", solvent_labels_excluding_neat),
        ],
        {
            "condition_statistics": table(stats, "Units: temperature °C, yield %, time h, loading mol%"),
            "solvent_categories": table(counts, "Figure 3 categories, mixtures grouped and xylenes pooled"),
            "source_solvents": table(raw, "All original solvent strings", evidence_kind="source_record"),
        },
        protocol="mannich.conditions.notebook.v1",
        notes=[
            EvidenceNote(
                code="SOLVENT_COUNT_DEFINITIONS",
                message="The article's 51 can be recovered "
                "as case-folded non-mixture solvent labels excluding neat, before pooling xylene labels. "
                "This is a label count, not a verified count of unique chemical solvent identities: "
                "synonyms such as DCM/CH2Cl2 are not automatically resolved. Figure3 categories use "
                "a different normalization and retain mixtures and neat explicitly.",
            ),
            EvidenceNote(
                code="TEMPERATURE_POLICY",
                message="The source records room temperature "
                "as 25°C and uses the initial temperature for ramps; original alternatives cannot be recovered "
                "from the normalized numeric column.",
            ),
        ],
    )


@lru_cache(maxsize=1)
def _rulebook():
    return json.loads((DATA_ROOT / "mannich/classification_rules.json").read_text())


def classify_molecule(smiles, role):
    rules = _rulebook()[role]
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    hits = []
    for rule in rules:
        matches = []
        for pattern in rule["patterns"]:
            # Dot-separated imine motifs mean AND in the deposited notebook.
            parts = [Chem.MolFromSmarts(p) for p in pattern.split(".")]
            if any(p is None for p in parts):
                raise ValueError("Invalid curated SMARTS")
            matches.append(all(mol.HasSubstructMatch(p) for p in parts))
        if any(matches):
            hits.append(rule["name"])
    if role == "organocatalyst":
        suppression = {
            "proline_pyrrolidine_derivatives": {"amino_acid", "1,2-diamine"},
            "imidazolidinone": {"amino_acid"},
            "cinchona_alkaloid": {"1,2-diamine"},
        }
        removed = set().union(*(suppression.get(h, set()) for h in hits))
        hits = [h for h in hits if h not in removed]
    return hits[0] if hits else "other"


@lru_cache(maxsize=1)
def _classified():
    df = load()
    for role in ("organocatalyst", "nucleophile", "electrophile"):
        lookup = {s: classify_molecule(s, role) for s in df[role].unique()}
        df[role + "_computed_class"] = df[role].map(lookup)
    unique = df.drop_duplicates("catalyst_identity")
    rare = set(unique.organocatalyst_computed_class.value_counts().loc[lambda s: s < 5].index) - {"other"}
    df.loc[df.organocatalyst_computed_class.isin(rare), "organocatalyst_computed_class"] = "other"
    return df


def component_classes():
    df = _classified().copy()
    rows = []
    class_counts = []
    for role in ("organocatalyst", "nucleophile", "electrophile"):
        identity = "catalyst_identity" if role == "organocatalyst" else role + "_canonical"
        classes = set(df[role + "_computed_class"])
        class_counts.append({"role": role, "classes_including_other": len(classes),
                             "named_classes_excluding_other": len(classes - {"other"}),
                             "unique_structures": int(df[identity].nunique()), "reaction_records": len(df)})
        for cls, group in df.groupby(role + "_computed_class"):
            rows.append(
                {
                    "role": role,
                    "class": cls,
                    "n_reactions": len(group),
                    "reaction_fraction": len(group) / len(df),
                    "n_unique_structures": group[identity].nunique(),
                }
            )
    mism = df.loc[
        df.organocatalyst_class != df.organocatalyst_computed_class,
        ["reaction_id", "organocatalyst_class", "organocatalyst_computed_class"],
    ]
    representatives = []
    for cls, group in df.groupby("organocatalyst_computed_class", sort=True):
        counts = group.groupby(["organocatalyst_canonical", "axial_configuration"], sort=True).size()
        maximum = int(counts.max())
        tied = counts.loc[counts == maximum]
        # Every tied maximum is retained; a lexicographic choice must not look like a chemical preference.
        for (smiles, axial), count in tied.items():
            representatives.append(
                {
                    "class": cls,
                    "smiles": smiles,
                    "axial_configuration": axial,
                    "n_reactions": int(count),
                    "n_tied_maxima": len(tied),
                }
            )
    catalyst_counts = pd.DataFrame(rows).query("role == 'organocatalyst'")
    dominant = catalyst_counts.sort_values(
        ["n_reactions", "class"], ascending=[False, True]
    ).head(4).copy()
    dominant["percent_of_all_reactions"] = 100 * dominant.n_reactions / len(df)
    return _result(
        "Ordered SMARTS assignments were recomputed from structures, with catalyst suppression "
        "rules and rare-class collapse evaluated on unique catalyst identities.",
        [_metric("stored_catalyst_class_disagreements", len(mism)), _metric("reaction_records", len(df)),
         _metric("four_dominant_catalyst_classes_records", int(dominant.n_reactions.sum())),
         _metric("four_dominant_catalyst_classes_percent", float(100 * dominant.n_reactions.sum() / len(df)), unit="%")],
        {
            "dominant_catalyst_classes": table(dominant, "Top four classes by REACTION count; denominator is all retained reaction records, not unique catalyst identities"),
            "class_count_summary": table(pd.DataFrame(class_counts),
                "Explicit class totals, including versus excluding other; structures and reaction records are different counts"),
            "class_frequencies": table(
                pd.DataFrame(rows), "Reaction and unique-structure counts are distinct"
            ),
            "label_disagreements": table(mism, "Deposited versus recomputed catalyst classes"),
            "class_representatives": table(
                pd.DataFrame(representatives),
                "Most frequent axial-aware catalyst identities per class; all tied maxima retained",
            ),
        },
        protocol="mannich.ordered_smarts.v1",
        rare_catalyst_class_minimum=5,
        notes=[
            EvidenceNote(
                code="MOTIF_CLASSIFICATION",
                message="These are the article's motif rules. "
                "Separate imine/substituent matches do not establish an atom-mapped reaction mechanism.",
            )
        ],
    )


def scaffold_diversity():
    df = load()
    rows = []
    summaries = []
    assignments = []
    for role in ("organocatalyst", "electrophile", "nucleophile"):
        key = "catalyst_identity" if role == "organocatalyst" else role + "_canonical"
        unique = df.drop_duplicates(key).copy()
        unique["scaffold"] = unique[role + "_canonical"].map(scaffold)
        for _, member in unique.iterrows():
            assignments.append({
                "role": role, "first_reaction_id": member.reaction_id,
                "canonical_smiles": member[role + "_canonical"],
                "axial_configuration": member.axial_configuration if role == "organocatalyst" else None,
                "scaffold_smiles": member.scaffold,
                "acyclic_excluded_from_scaffold_count": member.scaffold == "",
            })
        counts = Counter(unique.scaffold)
        reaction_counts = Counter(df[role + "_canonical"].map(scaffold))
        summaries.append(
            {
                "role": role,
                "unique_structures": len(unique),
                "acyclic_excluded": counts.get("", 0),
                "nonempty_scaffolds": len([s for s in counts if s]),
            }
        )
        for core, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if core:
                rows.append(
                    {
                        "role": role,
                        "scaffold_smiles": core,
                        "n_unique_structures": n,
                        "n_reactions": reaction_counts[core],
                    }
                )
    references = {"organocatalyst": 204, "electrophile": 114, "nucleophile": 112}
    comparisons = [
        {
            "role": r["role"],
            "computed_scaffolds": r["nonempty_scaffolds"],
            "paper_reported_scaffolds": references[r["role"]],
            "matches": r["nonempty_scaffolds"] == references[r["role"]],
        }
        for r in summaries
    ]
    discrepancies = [r for r in comparisons if not r["matches"]]
    version_audit = json.loads((DATA_ROOT / "mannich/scaffold_version_audit.json").read_text())
    source_hash = verified_manifest("mannich")["files"]["mannich_dataset.csv"]["sha256"]
    if version_audit["source_sha256"] != source_hash:
        raise ValueError("Scaffold version diagnostic refers to a different source table")
    notes = [
        EvidenceNote(
            code="SUPPLEMENT_SCAFFOLD_DISAGREEMENT",
            message=f"{r['role']}: computed {r['computed_scaffolds']} scaffolds, supplement reports "
            f"{r['paper_reported_scaffolds']}. No structures were dropped to force agreement.",
        )
        for r in discrepancies
    ]
    notes.append(EvidenceNote(
        code="SCAFFOLD_VERSION_CHECK", severity="info",
        message="Stored diagnostic runs on the same frozen table gave 207/114/112 with RDKit "
        "2025.09.2 (paper methods), 2025.09.6 (repository environment), and 2026.03.3. "
        "Individual scaffold strings also agreed. The tested versions do not explain 207 versus 204. "
        "These are previous diagnostic results, not fresh execution of three environments in this call. "
        "The largest-fragment calculation was a counterfactual only; production chemistry was unchanged.",
    ))
    return _result(
        "Achiral Bemis–Murcko scaffolds were recomputed separately for catalysts, electrophiles "
        "and nucleophiles. Acyclic molecules are reported separately and excluded from scaffold counts. "
        f"{len(discrepancies)} of three counts disagree with the supplement; reproduction is incomplete.",
        [],
        {
            "role_summary": table(pd.DataFrame(summaries), "Supplementary scaffold populations"),
            "scaffolds": table(pd.DataFrame(rows), "Unique-structure and reaction frequencies"),
            "scaffold_assignments": table(pd.DataFrame(assignments),
                "Every unique component identity, including acyclic exclusions; inspect membership before changing the reported 207/204 discrepancy"),
            "reference_comparison": table(
                pd.DataFrame(comparisons), "Computed and paper-reported columns are explicitly distinguished"
            ),
            "software_version_check": table(
                pd.DataFrame(version_audit["rows"]),
                "Stored version diagnostics on this exact dataset; not author results or fresh multi-version execution",
                evidence_kind="source_record",
            ),
        },
        protocol="mannich.scaffolds.v2",
        status="partial" if discrepancies else "success",
        notes=notes,
        remove_stereochemistry=True,
        acyclic_policy="exclude",
        version_diagnostic_run=version_audit["linux_run_url"],
        version_diagnostic_source="stored_own_reconstruction",
    )



def heatmap_protocol_comparison(result, variants):
    comparisons, changed = [], []
    for role in ("electrophile", "nucleophile"):
        columns = ("organocatalyst_class", role + "_computed_class")
        def indexed(mode):
            rows = variants[mode].answer.tables[role + "_heatmap"].rows
            index = {tuple(row[key] for key in columns): row for row in rows}
            if len(index) != len(rows):
                raise ValueError("Duplicate observed class pairs in protocol comparison")
            return index
        paper, notebook = indexed("paper"), indexed("notebook")
        if paper.keys() != notebook.keys():
            raise ValueError("Protocol comparison must preserve the observed class-pair population")
        differing = []
        for key, left in paper.items():
            right = notebook[key]
            if (left["n_reactions"], left["median_ee"]) != (right["n_reactions"], right["median_ee"]):
                raise ValueError("Masking protocols changed underlying observations")
            if left["visible"] != right["visible"]:
                differing.append({"substrate_role": role, "catalyst_class": key[0], "substrate_class": key[1],
                                  "n_reactions": left["n_reactions"], "median_ee": left["median_ee"],
                                  "paper_visible": left["visible"], "notebook_visible": right["visible"]})
        changed.extend(differing)
        comparisons.append({"substrate_role": role, "observed_class_pairs": len(paper),
                            "paper_minimum_n": variants["paper"].metadata.protocol_parameters["minimum_cell_count"],
                            "notebook_minimum_n": variants["notebook"].metadata.protocol_parameters["minimum_cell_count"],
                            "paper_visible_pairs": sum(row["visible"] for row in paper.values()),
                            "notebook_visible_pairs": sum(row["visible"] for row in notebook.values()),
                            "pairs_with_n4": sum(row["n_reactions"] == 4 for row in paper.values()),
                            "changed_visibility_pairs": len(differing)})
    result.answer.tables["masking_comparison"] = table(pd.DataFrame(comparisons),
        "Actual effect on observed class pairs; unobserved grid cells are not counted as zero observations")
    result.answer.tables["changed_mask_cells"] = table(pd.DataFrame(changed, columns=[
        "substrate_role", "catalyst_class", "substrate_class", "n_reactions", "median_ee", "paper_visible", "notebook_visible"]),
        "Only observed pairs whose visibility differs; an empty table means the masks agree on these data")
    n = sum(row["observed_class_pairs"] for row in comparisons)
    result.answer.metrics.extend([
        _metric("observed_class_pairs_compared", n, population="observed catalyst/substrate class pairs across both roles"),
        _metric("changed_visibility_pairs", len(changed), population="observed catalyst/substrate class pairs across both roles"),
    ])
    result.answer.finding += (f" The definitions differ, but {len(changed)} of {n} observed class pairs change visibility "
                             "on this actual dataset. Counts and median ee are identical before masking. "
                             "These observational medians do not establish the best catalyst for an unseen substrate.")
    return result


def selectivity_heatmaps(protocol="paper"):
    if protocol not in {"paper", "notebook"}:
        raise ValueError("protocol must be paper or notebook")
    df = _classified().copy()
    minimum = 5 if protocol == "paper" else 4
    tables = {}
    metrics = []
    mask_rows = []
    leaders = []
    for role in ("electrophile", "nucleophile"):
        # The deposited figure notebook uses the stored catalyst labels.
        grouped = (
            df.groupby(["organocatalyst_class", role + "_computed_class"], sort=True)
            .ee.agg(n_reactions="count", median_ee="median")
            .reset_index()
        )
        grouped["visible"] = grouped.n_reactions >= minimum
        grouped["display_median_ee"] = grouped.median_ee.where(grouped.visible, np.nan)
        tables[role + "_heatmap"] = table(
            grouped, "Observed class combinations; unseen combinations are absent, not zero-ee"
        )
        visible = grouped.loc[grouped.visible, ["organocatalyst_class", role + "_computed_class", "n_reactions", "median_ee"]].copy()
        visible = visible.sort_values(["median_ee", "n_reactions", "organocatalyst_class", role + "_computed_class"],
                                      ascending=[False, False, True, True])
        tables[role + "_visible_by_median"] = table(visible,
            "All visible observed class pairs ordered by median ee, then observation count; no universal high-ee threshold or causal recommendation")
        if not visible.empty:
            top = visible.loc[visible.median_ee == visible.median_ee.max()]
            leaders.append(role + " maximum visible median: " + "; ".join(
                f"{row['organocatalyst_class']} × {row[role + '_computed_class']}: {row['median_ee']:g}% ee, n={int(row['n_reactions'])}"
                for _, row in top.iterrows()))
        counts = {"observed_class_pairs": len(grouped), "pairs_with_n4": int((grouped.n_reactions == 4).sum()),
                  "visible_at_n_ge4": int((grouped.n_reactions >= 4).sum()), "visible_at_n_ge5": int((grouped.n_reactions >= 5).sum())}
        mask_rows.append({"substrate_role": role, **counts})
        metrics.extend(_metric(role + "_" + name, value, unit="class pairs", population=f"observed catalyst/{role} class pairs")
                       for name, value in counts.items())
    tables["mask_effect_by_role"] = table(pd.DataFrame(mask_rows),
        "Both masking thresholds evaluated on every observed pair, including rows beyond a preview")
    return _result(
        "Median ee and counts were computed for catalyst–substrate class pairs. Sparse cells "
        "are masked by the explicitly selected protocol; these observational medians are not "
        "causal estimates of catalyst superiority. " + " | ".join(leaders) + ". " +
        "Threshold4 versus5 changes only n=4 pairs; actual full-cohort counts for BOTH roles are in mask_effect_by_role.",
        metrics,
        tables,
        protocol=f"mannich.heatmap.{protocol}.v2",
        minimum_cell_count=minimum,
        notes=[
            EvidenceNote(
                code="PAPER_NOTEBOOK_MASK_DIFFERENCE",
                message="Figure 7 caption masks "
                "cells with fewer than 5 reactions, but the deposited notebook calls min_n=4. "
                "Both protocols are explicit; they must not be reported as identical.",
            )
        ],
    )
