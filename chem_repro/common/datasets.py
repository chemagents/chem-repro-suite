"""Pinned input registry shared across article adapters."""

import json
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .contracts import DataSource, Provenance, Table

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _source_sha256():
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


_IMPORTED_SOURCE_SHA256 = _source_sha256()


def implementation_sha256():
    current = _source_sha256()
    if current != _IMPORTED_SOURCE_SHA256:
        raise RuntimeError(
            "Package source changed after process startup; restart before producing scientific provenance"
        )
    return current


@lru_cache(maxsize=8)
def verified_manifest(study: str):
    if study not in {"mannich", "heracleum", "heracleum_paper", "cannabis", "cannabis_open_tox", "open_syntelly_2024", "cannabis_dmpnn", "cannabis_candidates"}:
        raise ValueError("Unknown study")
    root = DATA_ROOT / study
    manifest = json.loads((root / "manifest.json").read_text())
    for name, spec in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid manifest path")
        if sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Input checksum mismatch: {study}/{name}")
    return manifest


def provenance(study: str, protocol: str, **parameters) -> Provenance:
    m = verified_manifest(study)
    return Provenance(
        study_id=m["study_id"],
        paper_doi=m["paper_doi"],
        protocol_id=protocol,
        protocol_parameters=parameters,
        implementation_sha256=implementation_sha256(),
        sources=[
            DataSource(
                name=n,
                sha256=s["sha256"],
                source_url=s["source_url"],
                revision=s.get("revision", m["revision"]),
                derivation=s.get("derivation"),
                parent_sha256=s.get("parent_sha256"),
            )
            for n, s in m["files"].items()
        ],
        software={
            p: version(p)
            for p in ["chem-repro-suite", "rdkit", "numpy", "pandas", "scipy", "scikit-learn", "fastmcp"]
        },
    )


def table(frame: pd.DataFrame, description: str, evidence_kind="computed") -> Table:
    # pandas serializes missing cells as JSON null, retaining every row and declared column.
    return Table(
        description=description,
        columns=[str(c) for c in frame.columns],
        rows=json.loads(frame.to_json(orient="records", double_precision=15)),
        evidence_kind=evidence_kind,
    )
