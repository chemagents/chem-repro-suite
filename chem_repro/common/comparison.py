"""Compose declared variants without merging their populations or losing provenance."""
from __future__ import annotations

import re
from copy import deepcopy
from .contracts import AnalysisResult, Answer, Table


def compare_record_tables(reference: Table, observed: Table, *, id_column: str,
                          fields: list[str], reference_label="reference", observed_label="observed") -> Table:
    """Compare complete records by an explicit identity; never join previews or coerce IDs."""
    if reference.is_preview or observed.is_preview:
        raise ValueError("Record comparison requires complete tables, not previews")
    if not fields or len(fields) != len(set(fields)) or id_column in fields:
        raise ValueError("Comparison fields must be distinct and separate from identity")
    if reference_label == observed_label or any(not re.fullmatch(r"[a-z][a-z0-9_]*", x)
                                               for x in (reference_label, observed_label)):
        raise ValueError("Distinct explicit variant labels are required")
    indexed = []
    for value in (reference, observed):
        if not {id_column, *fields}.issubset(value.columns):
            raise ValueError("Comparison columns are missing")
        index = {}
        for row in value.rows:
            identity = row[id_column]
            if type(identity) not in (str, int) or identity == "" or identity in index:
                raise ValueError("Record identities must be unique nonempty strings or integers")
            index[identity] = row
        indexed.append(index)
    left, right = indexed
    rows = []
    for identity in dict.fromkeys([*left, *right]):
        a, b = left.get(identity), right.get(identity)
        row = {id_column: identity, reference_label + "__present": a is not None,
               observed_label + "__present": b is not None}
        for field in fields:
            row[reference_label + "__" + field] = deepcopy(a[field]) if a is not None else None
            row[observed_label + "__" + field] = deepcopy(b[field]) if b is not None else None
        row["changed_fields"] = ([field for field in fields if a[field] != b[field]]
                                 if a is not None and b is not None else ["record_presence"])
        rows.append(row)
    columns = [id_column, reference_label + "__present", observed_label + "__present"]
    columns += [label + "__" + field for field in fields for label in (reference_label, observed_label)]
    columns.append("changed_fields")
    return Table(description="Complete record comparison; null values and absent records remain distinct",
                 columns=columns, rows=rows)


def compare_variants(variants: dict[str, AnalysisResult], *, protocol_id: str, finding: str) -> AnalysisResult:
    if not variants or any(not re.fullmatch(r"[a-z][a-z0-9_]*", label) for label in variants):
        raise ValueError("Comparison requires nonempty, explicit variant labels")
    first = next(iter(variants.values())).metadata
    for result in variants.values():
        m = result.metadata
        if (m.study_id, m.paper_doi, m.implementation_sha256, m.software) != (
            first.study_id, first.paper_doi, first.implementation_sha256, first.software
        ):
            raise ValueError("Comparison variants must use the same study and implementation")
    statuses = {r.status for r in variants.values()}
    status = "success" if statuses == {"success"} else "unsupported" if statuses == {"unsupported"} else "partial"
    metrics, tables, notes, artifacts, sources, protocols = [], {}, [], [], {}, {}
    for label, result in variants.items():
        protocols[label] = result.metadata.model_dump(mode="json")
        for source in result.metadata.sources:
            key = (source.name, source.sha256, source.revision)
            if key in sources and sources[key] != source:
                raise ValueError("Conflicting provenance for one source identity")
            sources[key] = source.model_copy(deep=True)
        metrics.extend(m.model_copy(update={"name": f"{label}__{m.name}", "population": f"{label}: {m.population}"})
                       for m in result.answer.metrics)
        tables.update({f"{label}__{name}": table.model_copy(deep=True) for name, table in result.answer.tables.items()})
        notes.extend(n.model_copy(update={"code": f"{label.upper()}__{n.code}", "message": f"{label}: {n.message}"})
                     for n in result.answer.limitations)
        artifacts.extend(a.model_copy(update={"name": f"{label}__{a.name}"}) for a in result.artifacts)
    tables["variant_inventory"] = Table(
        description="Distinct executed variants; each retains its own parameters and population",
        columns=["variant", "protocol_id", "status", "finding"],
        rows=[{"variant": label, "protocol_id": r.metadata.protocol_id, "status": r.status,
               "finding": r.answer.finding} for label, r in variants.items()],
    )
    metadata = first.model_copy(update={
        "protocol_id": protocol_id, "protocol_parameters": {"variants": protocols},
        "sources": list(sources.values()), "reused_from_cache": all(r.metadata.reused_from_cache for r in variants.values()),
    }, deep=True)
    return AnalysisResult(status=status, answer=Answer(finding=finding, metrics=metrics, tables=tables, limitations=notes),
                          metadata=metadata, artifacts=artifacts)


def compare_identifier_sets(reference, observed, *, identity_definition):
    if not isinstance(identity_definition, str) or not identity_definition.strip():
        raise ValueError("Set agreement requires an explicit identity definition")
    values = []
    for identifiers in [reference, observed]:
        identifiers = list(identifiers)
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("Comparison identifiers must be nonempty strings")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate identifiers: define and deduplicate the comparison population explicitly")
        values.append(set(identifiers))
    expected, actual = values
    union = expected | actual
    return {
        "identity_definition": identity_definition,
        "reference_count": len(expected),
        "observed_count": len(actual),
        "intersection_count": len(expected & actual),
        "union_count": len(union),
        "jaccard": len(expected & actual) / len(union) if union else None,
        "reference_only": sorted(expected - actual),
        "observed_only": sorted(actual - expected),
    }
