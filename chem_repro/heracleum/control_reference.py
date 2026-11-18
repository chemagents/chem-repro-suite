"""Published control summaries are references, never new target-level test results."""

import json
import os

import pandas as pd

from ..common.artifacts import save_artifact
from ..common.contracts import EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest


def attach_control_reference(result):
    manifest = verified_manifest("heracleum_paper")
    root = DATA_ROOT / "heracleum_paper"
    reference = json.loads((root / "control_reference.json").read_text())
    parent = manifest["files"]["paper.pdf"]["sha256"]
    if reference["parent_sha256"] != parent:
        raise ValueError("Control reference belongs to a different paper")
    if "paper_reported_control_discrimination" in result.answer.tables:
        return result
    fields = [
        ("inactive_control_target_coverage", "targets", "paper control panel"),
        ("targets_with_sufficient_observations", "targets", "paper per-target active/inactive comparisons"),
        ("targets_active_better_significant", "targets", "39 paper comparisons with sufficient observations"),
        ("inactive_control_panel_median_kcal_mol", "kcal/mol", "paper inactive-control panel scores"),
        ("active_control_panel_median_kcal_mol", "kcal/mol", "paper known-active panel scores"),
    ]
    for name, unit, population in fields:
        result.answer.metrics.append(Metric(name="paper_" + name, value=reference[name], unit=unit,
                                            population=population, evidence_kind="paper_reported"))
    result.answer.tables["paper_reported_control_discrimination"] = table(
        pd.DataFrame([{key: value for key, value in reference.items() if key not in {"parent_sha256", "evidence_kind"}}]),
        "Main article sections2.3.1/4.4/4.8, PDF pages5/15/17: published panel medians and13/39 discriminating targets. "
        "One-sided Mann-Whitney with Bonferroni, not BH; global alpha0.05. Target-level observations/p-values are unavailable; "
        "no control test was recomputed. This denominator is independent of the42 redocking targets.",
        evidence_kind="paper_reported")
    result.answer.limitations.append(EvidenceNote(
        code="CONTROL_REFERENCE_NOT_RECOMPUTED",
        message="The published13/39 active-versus-inactive comparisons and panel medians are source references. "
                "Their raw per-target scores/sample counts/p-values are unavailable, so neither significance nor "
                "the identities of the13 targets can be independently reproduced. The remaining26 are not shown "
                "to have equal binding. Control discrimination and native-pose recovery are different validations."))
    result.metadata.sources.extend(provenance("heracleum_paper", "heracleum.controls_reference.v1").sources)
    result.metadata.protocol_parameters["control_reference"] = {
        "execution_kind": "paper_reported_reference_only", "source_section": reference["source_section"],
        "pdf_page_1based": reference["pdf_page_1based"], "paper_sha256": parent,
        "source_methods_section": reference["source_methods_section"],
        "methods_pdf_page_1based": reference["methods_pdf_page_1based"],
        "source_statistics_section": reference["source_statistics_section"],
        "statistics_pdf_page_1based": reference["statistics_pdf_page_1based"],
        "original_control_test_recomputed": False,
    }
    result.artifacts.append(save_artifact((root / "paper.pdf").read_bytes(), suffix="pdf",
        media_type="application/pdf", public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7342"),
        display_name="heracleum_paper_source_reference.pdf"))
    return result
