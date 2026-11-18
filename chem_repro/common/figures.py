"""Shared publication-size vector output; JSON tables remain the numerical evidence."""

from __future__ import annotations

import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import save_artifact
from .orx_figstyle import use_style


def initialize_style():
    use_style()
    plt.rcParams.update(
        {
            "svg.hashsalt": "chem-repro-suite-v1",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def export_figure(fig, name: str, *, port: int):
    """Export vector PDF/SVG and a PNG preview, preserving the actual plotted values."""
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
    artifacts = []
    try:
        for ext, mime in [("pdf", "application/pdf"), ("svg", "image/svg+xml"), ("png", "image/png")]:
            data = io.BytesIO()
            metadata = (
                {"CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else {"Date": None}
                if ext == "svg"
                else None
            )
            fig.savefig(data, format=ext, dpi=220, bbox_inches="tight", metadata=metadata)
            artifacts.append(
                save_artifact(
                    data.getvalue(),
                    suffix=ext,
                    media_type=mime,
                    public_base_url=base,
                    display_name=name + "." + ext,
                )
            )
    finally:
        plt.close(fig)
    return artifacts
