"""Declared scientific output schema, shared by every article adapter."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


IdentityPolicy = Annotated[
    Literal["strict", "published_snapshot"],
    Field(description=(
        "Choose published_snapshot for numerical reproduction of published results: retain the archived "
        "records and groups, while reporting source identity conflicts separately. Choose strict for "
        "source-consistent molecule interpretation: depending on the tool, conflicts are excluded from "
        "fitting or left unranked. Strict can change the analyzed population and does not reproduce the "
        "published cohort. Neither mode resolves conflicting molecular identities."
    )),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EvidenceNote(StrictModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class Metric(StrictModel):
    name: str
    value: float | int | str | bool | None
    unit: str | None = None
    population: str
    evidence_kind: Literal["computed", "paper_reported", "source_record"] = "computed"


class Table(StrictModel):
    description: str
    columns: list[str]
    rows: list[dict[str, JsonValue]]
    evidence_kind: Literal["computed", "paper_reported", "source_record"] = "computed"
    total_rows: int | None = Field(default=None, ge=0)
    is_preview: bool = False
    full_table_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def complete_rows(self):
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("Duplicate table columns")
        allowed = set(self.columns)
        if any(set(row) != allowed for row in self.rows):
            raise ValueError("Every table row must supply exactly the declared columns; use null for missing")
        json.dumps(self.rows, allow_nan=False)
        if self.total_rows is None:
            self.total_rows = len(self.rows)
        if self.is_preview:
            if self.total_rows <= len(self.rows) or self.full_table_artifact_sha256 is None:
                raise ValueError("A table preview needs its full row count and checksummed full artifact")
        elif self.total_rows != len(self.rows):
            raise ValueError("A complete table must contain exactly total_rows rows")
        return self


class Artifact(StrictModel):
    name: str
    media_type: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    download_url: str
    artifact: str  # Existing scientific MCP artifact collector compatibility.
    scope: Literal["http", "s3"] = "http"
    bucket: str | None = None
    s3_key: str | None = None
    presigned_url: str | None = None


class DataSource(StrictModel):
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str
    revision: str
    derivation: str | None = None
    parent_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class Provenance(StrictModel):
    study_id: str
    paper_doi: str
    protocol_id: str
    protocol_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    sources: list[DataSource]
    software: dict[str, str]
    implementation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reused_from_cache: bool = False


class Answer(StrictModel):
    finding: str
    metrics: list[Metric] = Field(default_factory=list)
    tables: dict[str, Table] = Field(default_factory=dict)
    limitations: list[EvidenceNote] = Field(default_factory=list)


class AnalysisResult(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["success", "partial", "unsupported"]
    answer: Answer
    metadata: Provenance
    artifacts: list[Artifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def finite_json(self):
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self
