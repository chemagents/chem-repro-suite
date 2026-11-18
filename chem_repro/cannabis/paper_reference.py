"""Reported primary performance, separate from S4 and newly reconstructed validation."""

import pandas as pd

from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, provenance, table


def attach_primary_validation(output, models):
    source = provenance("cannabis_candidates", "cannabis.primary_performance_transcription.v1")
    frame = pd.read_csv(DATA_ROOT / "cannabis_candidates/paper_primary_validation.csv")
    if not models or set(models) - set(frame.model):
        raise ValueError("Unknown primary-performance model reference")
    output.answer.tables["paper_primary_reference"] = table(
        frame.loc[frame.model.isin(models)].reset_index(drop=True),
        "Main-text means and SD, not SupplementS4 or recomputed validation. Raw percent FPR and its fraction conversion are both retained.",
        "paper_reported",
    )
    present = {(s.name, s.sha256) for s in output.metadata.sources}
    output.metadata.sources.extend(s for s in source.sources
        if s.name in {"paper.pdf", "paper_primary_validation.csv"} and (s.name, s.sha256) not in present)
    output.answer.limitations.append(EvidenceNote(
        code="PRIMARY_REFERENCE_NOT_PAIRED_REPLAY",
        message="Main-text means and SD are separately reported references. Section4.7 describes aggregation across CV folds; "
        "the exact author partitions/checkpoints are unavailable. Do not relabel paired five-fold OOF values, a new source repeated-holdout "
        "mean, or SupplementS4 values as this primary result. The reported standardized reference total3403 is not proof of the authors' "
        "exact DMPNN training membership; this source implementation uses3171 labeled records.",
    ))
    return output
