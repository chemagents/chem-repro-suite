"""Checksummed artifacts served by the MCP process; no inaccessible remote paths."""

from __future__ import annotations

import json
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path

from .contracts import Artifact

_FILE = re.compile(r"^[0-9a-f]{64}\.(json|csv|png|svg|pdf|sdf|zip)$")


def artifact_root() -> Path:
    root = Path(os.getenv("CHEM_REPRO_ARTIFACTS_DIR", "artifacts")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_artifact(
    payload: bytes, *, suffix: str, media_type: str, public_base_url: str, display_name: str
) -> Artifact:
    digest = sha256(payload).hexdigest()
    filename = digest + "." + suffix
    if not _FILE.fullmatch(filename):
        raise ValueError("Unsupported artifact type")
    destination = artifact_root() / filename
    # Atomic publication: a failed producer never exposes half a file.
    if not destination.exists():
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
                temp = Path(f.name)
                f.write(payload)
            temp.replace(destination)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    elif sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError("Existing content-addressed artifact is corrupt")
    url = public_base_url.rstrip("/") + "/files/" + filename
    return Artifact(
        name=display_name,
        media_type=media_type,
        sha256=digest,
        size_bytes=len(payload),
        download_url=url,
        artifact=url,
    )


def save_json(data, *, public_base_url: str, display_name: str) -> Artifact:
    body = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return save_artifact(
        body,
        suffix="json",
        media_type="application/json",
        public_base_url=public_base_url,
        display_name=display_name,
    )


def save_table(data, *, port, display_name):
    import pandas as pd

    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
    rows = [
        {
            key: json.dumps(value, ensure_ascii=False, allow_nan=False)
            if isinstance(value, (dict, list))
            else value
            for key, value in row.items()
        }
        for row in data.rows
    ]
    # Keep the schema's Python scalars. Inference otherwise promotes nullable
    # integer columns to float64, changing 1 to 1.0 and rounding large IDs.
    body = pd.DataFrame(rows, columns=data.columns, dtype=object).to_csv(index=False).encode()
    return save_artifact(
        body,
        suffix="csv",
        media_type="text/csv",
        public_base_url=base,
        display_name=display_name,
    )


def export_tables(result, *, port, preview_rows=20, compact_max_rows=64, inline_char_budget=10000):
    """Export all rows; preserve compact summaries and bound larger previews explicitly."""
    if not isinstance(preview_rows, int) or preview_rows < 1:
        raise ValueError("preview_rows must be a positive integer")
    if not isinstance(compact_max_rows, int) or compact_max_rows < 0:
        raise ValueError("compact_max_rows must be a nonnegative integer")
    if not isinstance(inline_char_budget, int) or inline_char_budget < 2:
        raise ValueError("inline_char_budget must be at least two characters")
    for name, data in result.answer.tables.items():
        if data.is_preview:
            if not any(a.sha256 == data.full_table_artifact_sha256 for a in result.artifacts):
                raise ValueError("Preview has no full artifact in this result")
            continue
        artifact = save_table(data, port=port, display_name=result.metadata.protocol_id + "_" + name + ".csv")
        if not any(a.sha256 == artifact.sha256 and a.name == artifact.name for a in result.artifacts):
            result.artifacts.append(artifact)
        encode = lambda rows: json.dumps(rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        compact = len(data.rows) <= compact_max_rows and len(encode(data.rows)) <= inline_char_budget
        visible_rows = data.rows if compact else data.rows[:preview_rows]
        while visible_rows and len(encode(visible_rows)) > inline_char_budget:
            visible_rows = visible_rows[:-1]
        if len(visible_rows) < len(data.rows):
            result.answer.tables[name] = type(data).model_validate({
                **data.model_dump(), "rows": visible_rows, "total_rows": len(data.rows),
                "is_preview": True, "full_table_artifact_sha256": artifact.sha256,
            })
    return result


def attach_downloads(mcp):
    from starlette.requests import Request
    from starlette.responses import FileResponse, Response

    @mcp.custom_route("/files/{name}", methods=["GET"])
    async def download(request: Request):
        name = request.path_params["name"]
        if not _FILE.fullmatch(name):
            return Response(status_code=404)
        path = artifact_root() / name
        if not path.is_file() or path.is_symlink():
            return Response(status_code=404)
        # Verify against filename before delivering an artifact to a downstream experiment.
        if sha256(path.read_bytes()).hexdigest() != name.split(".")[0]:
            return Response("Artifact checksum mismatch", status_code=409)
        return FileResponse(path)
