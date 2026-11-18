"""Reanalysis of the published per-target redocking observations (Table S1a)."""

import pandas as pd

from ..common.contracts import AnalysisResult, Answer, EvidenceNote, Metric
from ..common.datasets import DATA_ROOT, provenance, table, verified_manifest
from ..common.docking import pose_recovery_summary


def protocol_validation():
    verified_manifest("heracleum")
    records = pd.read_csv(DATA_ROOT / "heracleum" / "redocking_table_s1a.csv")
    recovery, failed, excluded, median = pose_recovery_summary(records, rmsd_threshold=2.0)
    n = int(records.evaluated.sum())
    return AnalysisResult(
        status="partial",
        answer=Answer(
            finding=(
                f"Published pose-recovery observations yield {recovery.iloc[0].n_recovered:g}/{n} "
                f"top-1 and {recovery.iloc[-1].n_recovered:g}/{n} top-5 recovery at RMSD <= 2 Å. "
                f"Median best-top5 RMSD is {median:.2f} Å. Six targets fail; two are unevaluable. "
                "This is a recomputed summary of reported observations, not new redocking."
            ),
            metrics=[
                Metric(name="panel_targets", value=len(records), population="Table S1a panel"),
                Metric(name="evaluated_targets", value=n, population="Table S1a panel"),
                Metric(name="excluded_targets", value=len(excluded), population="Table S1a panel"),
                Metric(name="failed_top5_targets", value=len(failed), population="42 evaluated targets"),
                Metric(
                    name="median_best_top5_rmsd",
                    value=median,
                    unit="angstrom",
                    population="42 evaluated targets",
                ),
            ],
            tables={
                "pose_recovery": table(
                    recovery, "Aggregated source ranks; denominator excludes only two unevaluable targets"
                ),
                "target_observations": table(
                    records, "Transcribed Table S1a; RMSD and ranks were reported by authors", "source_record"
                ),
                "failed_targets": table(
                    failed, "No near-native pose among five source poses", "source_record"
                ),
                "excluded_targets": table(
                    excluded, "Not docking failures; unavailable native ligand/reference", "source_record"
                ),
            },
            limitations=[
                EvidenceNote(
                    code="REPORTED_POSE_OBSERVATIONS",
                    message="Original poses, atom correspondence, "
                    "receptor preparation and per-pose RMSD values are not bundled. No docking engine or "
                    "coordinate-based RMSD calculation was executed. Threshold cannot be changed from 2 Å "
                    "using first-recovery ranks alone.",
                ),
                EvidenceNote(
                    code="CONTROL_RECORDS_MISSING",
                    message="Raw active/inactive docking control "
                    "observations are not in Table S1a. The main-text 13/39 discrimination result is not "
                    "independently recomputed by this tool.",
                ),
                EvidenceNote(
                    code="POSE_NOT_AFFINITY",
                    message="Native-pose recovery does not validate affinity, "
                    "target inhibition or toxicity; it tests one part of docking protocol reliability.",
                ),
            ],
        ),
        metadata=provenance(
            "heracleum",
            "heracleum.published_pose_recovery.v1",
            rmsd_threshold_angstrom=2.0,
            maximum_rank=5,
            source_table="S1a",
            rmsd_definition="reported symmetry-aware fixed-frame heavy-atom best-top5 RMSD",
            execution_kind="aggregation_of_source_observations",
        ),
    )
