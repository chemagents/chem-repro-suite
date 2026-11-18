"""Heracleum figures drawn from actual source computations and saved OOF evidence."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.datasets import implementation_sha256
from ..common.figures import export_figure, initialize_style
from ..common.orx_figstyle import PALETTE
from ..common.structure_figures import structure_pages
from .analysis import docking_components
from .data import CLASSES, ENDPOINTS, SATURATED, load

SCAFFOLD_COLORS = {
    "Other": "#bdbdbd",
    "Furo 1": "#df65b0",
    "Furo 2": "#9370db",
    "Top 1": "#c44e52",
    "Top 2": "#377eb8",
    "Top 3": "#55a868",
}


def attach(result, fig, name):
    result.metadata.protocol_parameters["figure_renderer_implementation_sha256"] = implementation_sha256()
    result.artifacts.extend(export_figure(fig, name, port=7342))
    return result


def chemical_space(result):
    for subset in ["DS1", "DS2"]:
        rows = [r for r in result.answer.tables["top_scaffolds"].rows if r["subset"] == subset]
        result.artifacts.extend(
            structure_pages(
                rows,
                smiles_column="scaffold_smiles",
                legend_fn=lambda r: f"{r['rank']}. n={r['n_molecules']} ({r['fraction']:.1%})",
                name="heracleum_figure01_" + subset,
                port=7342,
                page_size=18,
                columns=3,
            )
        )
    return result


def candidate_structures(result):
    master, _ = load()
    lookup = master.set_index("synID").canonical_smiles
    rows = [
        dict(row, prediction_smiles=lookup.loc[row["synID"]])
        for row in result.answer.tables["candidates"].rows
        if row["tier"] == 1
    ]
    policy = result.metadata.protocol_parameters["identity_policy"]
    if rows:
        result.artifacts.extend(
            structure_pages(
                rows,
                smiles_column="prediction_smiles",
                legend_fn=lambda r: (
                    f"synID {r['synID']}; favorable {r['n_favorable']}/8\n"
                    + ("identity verified" if r["identity_verified"] else "IDENTITY UNRESOLVED")
                ),
                name="heracleum_tier1_" + policy + "_prediction_structures",
                port=7342,
            )
        )
    return result


def docking_profile(result):
    initialize_style()
    valid, _targets, matrix, pca, _ = docking_components()
    n_ds2 = int(valid.ds2_member.sum())
    fig, ax = plt.subplots(figsize=(6.75, 4.5), layout="constrained")
    for mask, color, label, marker, size in [
        (~valid.ds2_member, "#bdbdbd", f"DS1 only (n={len(valid) - n_ds2})", ".", 8),
        (valid.ds2_member, PALETTE["red"], f"DS2 including overlap (n={n_ds2})", "o", 14),
    ]:
        ax.scatter(
            valid.loc[mask, "PC1"],
            valid.loc[mask, "PC2"],
            c=color,
            marker=marker,
            s=size,
            alpha=0.7,
            linewidths=0.3,
            label=label,
        )
    ax.set(
        xlabel=f"PC1 ({pca['explained_variance_ratio'][0]:.1%} variance)",
        ylabel=f"PC2 ({pca['explained_variance_ratio'][1]:.1%} variance)",
    )
    ax.legend(fontsize=7)
    attach(result, fig, "heracleum_figure02_docking_pca")

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.0), layout="constrained")
    variance = pca["explained_variance_ratio"][:10]
    axes[0].bar(range(1, 11), variance, color=PALETTE["blue"], label="Individual")
    axes[0].plot(
        range(1, 11), np.cumsum(variance), "o-", color=PALETTE["red"], markersize=3, label="Cumulative"
    )
    axes[0].axhline(0.8, color="gray", ls="--", lw=0.7)
    axes[0].set(xlabel="Component", ylabel="Explained variance fraction", ylim=(0, 1.05))
    axes[0].legend(fontsize=6)
    axes[1].scatter(matrix.mean(axis=1), valid.PC1, s=4, alpha=0.3, color=PALETTE["blue"])
    axes[1].set(xlabel="Mean docking score (kcal/mol)", ylabel="PC1 score")
    attach(result, fig, "heracleum_supplement_scree_mean_pc1")

    fig, axes = plt.subplots(4, 3, figsize=(6.75, 8.5), layout="constrained")
    mean = matrix.mean(axis=1)
    for ax, (endpoint, column) in zip(axes.flat, ENDPOINTS.items()):
        for cls in ["Other", *CLASSES]:
            mask = valid.scaffold_class == cls
            ax.scatter(
                mean[mask], valid.loc[mask, column], s=3, alpha=0.35, linewidths=0, color=SCAFFOLD_COLORS[cls]
            )
        ax.scatter(
            mean[valid.ds2_member],
            valid.loc[valid.ds2_member, column],
            s=8,
            facecolors="none",
            edgecolors="#333333",
            linewidths=0.3,
        )
        ax.set(xlabel="Mean score (kcal/mol)", ylabel=endpoint.replace("_", " "), ylim=(-0.03, 1.03))
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[10:]:
        ax.set_axis_off()
    handles = [
        plt.Line2D(
            [],
            [],
            color=SCAFFOLD_COLORS[c],
            marker="o",
            linestyle="",
            markersize=4,
            label=f"{c} (n={int((valid.scaffold_class == c).sum())})",
        )
        for c in ["Other", *CLASSES]
    ]
    axes.flat[10].legend(handles=handles, loc="center", fontsize=7)
    attach(result, fig, "heracleum_supplement_docking_vs_toxicity_valid_cohort")

    strong = (matrix < -10).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.75, 3.4), layout="constrained")
    groups = [*CLASSES, "Other"]
    ax.boxplot(
        [strong[valid.scaffold_class == c] for c in groups],
        tick_labels=groups,
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.25},
    )
    ax.set(xlabel="Scaffold class (valid docking cohort)", ylabel="Targets with score < −10 kcal/mol (count)")
    attach(result, fig, "heracleum_supplement_promiscuity")
    result.metadata.protocol_parameters["scatter_population"] = (
        "2055 valid docking profiles; caption counts recomputed for plotted rows"
    )
    return result


def toxicity_landscape(result):
    initialize_style()
    frame, _ = load()
    names = list(ENDPOINTS)
    values = frame[list(ENDPOINTS.values())]
    fig, axes = plt.subplots(2, 1, figsize=(6.75, 8.5), layout="constrained")
    parts = axes[0].violinplot([values[c].to_numpy() for c in values], vert=False, showmedians=True)
    for name, body in zip(names, parts["bodies"]):
        body.set_facecolor(PALETTE["red"] if name in SATURATED else PALETTE["blue"])
        body.set_alpha(0.6)
    axes[0].set_yticks(range(1, 11), [n.replace("_", " ") + (" *" if n in SATURATED else "") for n in names])
    axes[0].invert_yaxis()
    axes[0].axvline(0.9, color="gray", ls="--", lw=0.7)
    axes[0].set(xlabel="ADMETlab predicted endpoint value (0–1)", xlim=(-0.03, 1.03))
    cor = values.corr(method="spearman").to_numpy()
    image = axes[1].imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].grid(False)
    axes[1].set_xticks(range(10), [n.replace("_", " ") for n in names], rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(10), [n.replace("_", " ") for n in names], fontsize=7)
    fig.colorbar(image, ax=axes[1], label="Spearman correlation", shrink=0.8)
    attach(result, fig, "heracleum_figure03_toxicity_landscape")
    variance = pd.DataFrame(result.answer.tables["pca_variance"].rows).variance_fraction.to_numpy()
    fig, ax = plt.subplots(figsize=(6.75, 3.3), layout="constrained")
    ax.bar(range(1, 11), variance, color=PALETTE["green"], label="Individual")
    ax.plot(range(1, 11), np.cumsum(variance), "o-", color=PALETTE["red"], label="Cumulative", markersize=3)
    ax.axhline(0.8, color="gray", ls="--", lw=0.7)
    ax.set(xlabel="Component", ylabel="Explained variance fraction", ylim=(0, 1.05))
    ax.legend(fontsize=7)
    return attach(result, fig, "heracleum_supplement_toxicity_scree")


def separability(result):
    initialize_style()
    rows = sorted(result.answer.tables["axis_tests"].rows, key=lambda r: r["observed_ks"])
    fig, ax = plt.subplots(figsize=(6.75, 4.6), layout="constrained")
    for i, row in enumerate(rows):
        ax.plot(
            [row["null_low95"], row["null_high95"]], [i, i], lw=5, color="#cccccc", solid_capstyle="round"
        )
        ax.scatter(
            row["observed_ks"],
            i,
            color=PALETTE["red"] if row["axis"].startswith("PC") else PALETTE["blue"],
            s=25,
        )
    ax.set_yticks(range(len(rows)), [r["axis"].replace("_", " ") for r in rows], fontsize=8)
    ax.set_xlabel("Mean pairwise KS distance (gray: 95% null interval)")
    ax.set_xlim(0, 1)
    return attach(result, fig, "heracleum_figure04_scaffold_separability")


def transfer(result):
    initialize_style()
    rows = result.answer.tables["validation_metrics"].rows
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")
    for i, row in enumerate(rows):
        ax.errorbar(
            i,
            row["r2"],
            yerr=[[row["r2"] - row["r2_ci_low"]], [row["r2_ci_high"] - row["r2"]]],
            fmt="o",
            capsize=5,
            color=PALETTE["blue"] if row["scheme"] == "random" else PALETTE["red"],
        )
    ax.set_xticks(range(len(rows)), [r["scheme"] for r in rows])
    ax.set(
        xlabel="Five-fold validation scheme",
        ylabel="Pooled OOF R² (95% bootstrap interval)",
        ylim=(-0.05, 1),
        xlim=(-0.6, 1.6),
    )
    parameters = result.metadata.protocol_parameters
    name = (
        "heracleum_transfer_"
        + parameters["model"]
        + "_"
        + parameters["endpoint"]
        + "_"
        + parameters["identity_policy"]
    )
    return attach(result, fig, name)


def transfer_comparison(results):
    """Combined Figure 5 from four completed analysis results; never refit for rendering."""
    initialize_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.4), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["RF", "ElasticNet"]):
        for endpoint_index, endpoint in enumerate(["neurotoxicity", "eye_corrosion"]):
            result = results[(model, endpoint)]
            for row in result.answer.tables["validation_metrics"].rows:
                random = row["scheme"] == "random"
                x = endpoint_index + (-0.17 if random else 0.17)
                ax.bar(
                    x,
                    row["r2"],
                    width=0.3,
                    yerr=[[row["r2"] - row["r2_ci_low"]], [row["r2_ci_high"] - row["r2"]]],
                    capsize=3,
                    color=PALETTE["blue"] if random else PALETTE["red"],
                    label=row["scheme"] if endpoint_index == 0 else None,
                )
        ax.set_xticks([0, 1], ["Neurotoxicity", "Eye corrosion"], fontsize=8)
        ax.set_xlabel(model)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Pooled OOF R² (95% bootstrap interval)")
    return export_figure(fig, "heracleum_figure05_all_models", port=7342)
