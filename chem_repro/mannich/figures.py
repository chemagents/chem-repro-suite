"""Article figures drawn from recomputed records, never screenshots of the original."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.chemistry import ee_to_ddg
from ..common.figures import export_figure, initialize_style
from ..common.orx_figstyle import PALETTE
from .analysis import load


def target_distribution(result):
    initialize_style()
    df = load()
    ddg = np.array([ee_to_ddg(e, t) for e, t in zip(df.ee, df.temperature)])
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), layout="constrained")
    for ax, values, bins, label, color, letter in [
        (axes[0], df.ee.to_numpy(), np.arange(0, 102, 2), "Enantiomeric excess (%)", PALETTE["blue"], "a"),
        (
            axes[1],
            ddg,
            np.arange(0, ddg.max() + 0.21, 0.2),
            r"$|\Delta\Delta G^\ddagger|$ (kcal/mol)",
            PALETTE["orange"],
            "b",
        ),
    ]:
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(np.mean(values), color="#333333", linestyle="--", linewidth=1, label="Mean")
        ax.axvline(np.median(values), color=PALETTE["red"], linestyle=":", linewidth=1, label="Median")
        ax.set_xlabel(label)
        ax.set_ylabel("Reaction records (count)")
        ax.set_ylim(bottom=0)
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes, fontweight="bold")
    axes[1].legend(fontsize=7, loc="upper right")
    result.artifacts.extend(export_figure(fig, "mannich_figure02_recomputed", port=7343))
    return result


def reaction_conditions(result):
    initialize_style()
    df = load()
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 6.2), layout="constrained")
    for ax, col, label, bins, color, letter in [
        (axes[0, 0], "temperature", "Temperature (°C)", np.arange(-80, 91, 5), PALETTE["blue"], "a"),
        (axes[0, 1], "yield", "Yield (%)", np.arange(0, 106, 5), PALETTE["green"], "b"),
        (axes[1, 1], "time", "Reaction time (h)", np.arange(0, 368, 8), PALETTE["orange"], "d"),
    ]:
        values = df[col].dropna().to_numpy()
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(np.mean(values), color="#333333", linestyle="--", linewidth=1, label="Mean")
        ax.axvline(np.median(values), color=PALETTE["red"], linestyle=":", linewidth=1, label="Median")
        ax.set_xlabel(label)
        ax.set_ylabel("Reaction records (count)")
        ax.set_ylim(bottom=0)
        ax.text(-0.15, 1.03, letter, transform=ax.transAxes, fontweight="bold")
    axes[1, 1].set_xlim(0, 360)
    axes[0, 0].legend(fontsize=7)
    ax = axes[1, 0]
    rows = result.answer.tables["solvent_categories"].rows
    chosen = rows[:12]
    remaining = sum(r["n_reactions"] for r in rows[12:])
    names = [r["solvent_category"] for r in chosen]
    values = [r["n_reactions"] for r in chosen]
    if remaining:
        names.append("other categories")
        values.append(remaining)
    y = np.arange(len(names))
    ax.barh(y, values, color=PALETTE["purple"])
    ax.set_yticks(y, names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Reaction records (count)")
    ax.set_ylabel("Solvent category")
    ax.text(-0.15, 1.03, "c", transform=ax.transAxes, fontweight="bold")
    result.artifacts.extend(export_figure(fig, "mannich_figure03_recomputed", port=7343))
    return result


def chemical_space(result, population):
    initialize_style()
    df = pd.DataFrame(result.answer.tables["coordinates"].rows)
    group = "class" if population == "catalysts" else "role"
    categories = sorted(df[group].unique())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(categories)))
    if population == "catalysts":
        fig, (ax, bars) = plt.subplots(2, 1, figsize=(6.75, 8), layout="constrained")
    else:
        fig, ax = plt.subplots(figsize=(6.75, 4.8), layout="constrained")
    for label, color in zip(categories, colors):
        part = df.loc[df[group] == label]
        ax.scatter(
            part.tsne_x,
            part.tsne_y,
            s=13,
            alpha=0.8,
            color=color,
            linewidths=0,
            label=f"{label.replace('_', ' ')} (n={len(part)})",
        )
    ax.set(xlabel="t-SNE 1 (arbitrary units)", ylabel="t-SNE 2 (arbitrary units)")
    if population == "catalysts":
        frequencies = df.groupby("class").n_reactions.sum().sort_values(ascending=False)
        palette = dict(zip(categories, colors))
        bars.barh(range(len(frequencies)), frequencies, color=[palette[c] for c in frequencies.index])
        bars.set_yticks(
            range(len(frequencies)), [fill(c.replace("_", " "), 28) for c in frequencies.index], fontsize=7
        )
        bars.invert_yaxis()
        bars.set_xlim(0, frequencies.max() * 1.16)
        for i, (label, n) in enumerate(frequencies.items()):
            bars.text(n + frequencies.max() * 0.02, i, str(n), va="center", fontsize=7)
        bars.set_xlabel("Reaction records (count)")
        bars.set_ylabel("Catalyst class (colors shared with panel a)")
        ax.text(-0.13, 1.02, "a", transform=ax.transAxes, fontweight="bold")
        bars.text(-0.13, 1.02, "b", transform=bars.transAxes, fontweight="bold")
    else:
        ax.legend(loc="best", fontsize=7)
    result.artifacts.extend(export_figure(fig, f"mannich_{population}_space", port=7343))
    return result


def selectivity_heatmaps(result):
    initialize_style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), layout="constrained")
    for ax, role in zip(axes, ["electrophile", "nucleophile"]):
        frame = pd.DataFrame(result.answer.tables[role + "_heatmap"].rows)
        values = frame.pivot(
            index="organocatalyst_class", columns=role + "_computed_class", values="display_median_ee"
        )
        counts = frame.pivot(
            index="organocatalyst_class", columns=role + "_computed_class", values="n_reactions"
        )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(values.to_numpy(float), cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(
            range(len(values.columns)),
            [v.replace("_", " ") for v in values.columns],
            rotation=55,
            ha="right",
            fontsize=7,
        )
        ax.set_yticks(range(len(values)), [v.replace("_", " ") for v in values.index], fontsize=7)
        ax.set_title(role.capitalize() + " (median ee; observation count below)")
        ax.grid(False)
        for i in range(len(values)):
            for j in range(len(values.columns)):
                value = values.iloc[i, j]
                if pd.notna(value):
                    ax.text(
                        j,
                        i,
                        f"{value:g}\n{int(counts.iloc[i, j])}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="black" if value > 55 else "white",
                    )
        fig.colorbar(im, ax=ax, label="Median ee (%)", shrink=0.65)
    result.artifacts.extend(
        export_figure(fig, "mannich_figure07_" + result.metadata.protocol_id.split(".")[-2], port=7343)
    )
    return result


def representative_structures(result):
    from ..common.structure_figures import structure_pages

    records = result.answer.tables["class_representatives"].rows
    labels = {
        "urea_thiourea_guanidine_isothiourea": "urea/thiourea/guanidine/isothiourea",
        "proline_pyrrolidine_derivatives": "proline/pyrrolidine derivatives",
    }

    def legend(row):
        label = labels.get(row["class"], row["class"].replace("_", " "))
        return f"{label}\nn={row['n_reactions']}; axial annotation: {row['axial_configuration']}"

    result.artifacts.extend(
        structure_pages(
            records,
            smiles_column="smiles",
            legend_fn=legend,
            name="mannich_figure05_representatives",
            port=7343,
            page_size=12,
            columns=3,
        )
    )
    return result


def scaffold_structures(result):
    from ..common.structure_figures import structure_pages

    records = result.answer.tables["scaffolds"].rows
    for role in ["organocatalyst", "electrophile", "nucleophile"]:
        rows = [dict(r, rank=i + 1) for i, r in enumerate(r for r in records if r["role"] == role)]
        result.artifacts.extend(
            structure_pages(
                rows,
                smiles_column="scaffold_smiles",
                legend_fn=lambda r: (
                    f"{r['rank']}  ({r['n_unique_structures']} structures; {r['n_reactions']} reactions)"
                ),
                name="mannich_scaffolds_" + role,
                port=7343,
            )
        )
    return result


def reaction_pca(result, coordinates):
    initialize_style()
    fig, ax = plt.subplots(figsize=(6.75, 4.6), layout="constrained")
    ax.scatter(coordinates.PC1, coordinates.PC2, s=5, alpha=0.25, linewidths=0, color=PALETTE["blue"])
    fractions = result.answer.tables["explained_variance"].rows
    ax.set_xlabel(f"PC1 ({fractions[0]['fraction']:.1%} variance)")
    ax.set_ylabel(f"PC2 ({fractions[1]['fraction']:.1%} variance)")
    name = "mannich_reaction_pca_agents_" + str(result.metadata.protocol_parameters["include_agents"])
    result.artifacts.extend(export_figure(fig, name, port=7343))
    return result
