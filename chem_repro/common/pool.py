"""Single tool-contract and transport implementation used by all three study servers."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
from collections.abc import Callable
from functools import wraps
from typing import get_type_hints

from fastmcp import FastMCP
from fastmcp.tools import Tool
from mcp.types import ToolAnnotations

from .artifacts import attach_downloads, export_tables
from .contracts import AnalysisResult


def study_server(
    name: str, doi: str, *, artifact_port: int | None = None, operational_numeric_defaults: bool = False
) -> FastMCP:
    mcp = FastMCP(
        name,
        instructions=(
            f"Quantitative and qualitative analysis of {doi}. Return computed metrics and their "
            "population/method, distinguish paper-reported references and manual curation, retain source "
            "conflicts. A result inconsistent with the article is not a reason to change inputs or methods. "
            "CoScientist owns planning, task state, artifact capture and final narrative."
        ),
    )
    attach_downloads(mcp)
    mcp._chem_repro_artifact_port = artifact_port
    mcp._chem_repro_paper_doi = doi
    mcp._chem_repro_operational_numeric_defaults = operational_numeric_defaults
    return mcp


def register(
    mcp: FastMCP,
    fn: Callable,
    *,
    name: str,
    dataset_scope: str,
    input_mode: str = "fixed_dataset",
    read_only: bool = True,
):
    if input_mode not in {"fixed_dataset", "caller_data"}:
        raise ValueError("Tool input mode must be explicit")
    lock = getattr(mcp, "_chem_repro_execution_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        mcp._chem_repro_execution_lock = lock

    def finish(value):
        port = getattr(mcp, "_chem_repro_artifact_port", None)
        if isinstance(value, AnalysisResult) and port is not None:
            return export_tables(value, port=port)
        return value

    def synchronous_call(*args, **kwargs):
        return finish(fn(*args, **kwargs))

    @wraps(fn)
    async def responsive_call(*args, **kwargs):
        # Keep scientific operations serialized (including matplotlib state),
        # but leave the MCP event loop available for discovery and downloads.
        async with lock:
            if inspect.iscoroutinefunction(fn):
                return await asyncio.to_thread(finish, await fn(*args, **kwargs))
            work = asyncio.create_task(asyncio.to_thread(synchronous_call, *args, **kwargs))
            cancelled = False
            while True:
                try:
                    value = await asyncio.shield(work)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if work.done():
                        break
            if cancelled:
                if not work.cancelled():
                    work.exception()
                raise asyncio.CancelledError
            return value

    hints = get_type_hints(fn, include_extras=True)
    signature = inspect.signature(fn)
    responsive_call.__signature__ = signature.replace(
        parameters=[parameter.replace(annotation=hints.get(name, parameter.annotation))
                    for name, parameter in signature.parameters.items()],
        return_annotation=hints.get("return", signature.return_annotation),
    )
    responsive_call.__annotations__ = hints
    tool = Tool.from_function(
        responsive_call,
        name=name,
        annotations=ToolAnnotations(
            readOnlyHint=read_only, destructiveHint=False, idempotentHint=read_only, openWorldHint=False
        ),
    )
    # CoScientist indexes tool descriptions; server instructions alone are not in that index.
    # Keep the paper and population visible when similar analyses exist across article adapters.
    tool.description = (
        f"Study: {mcp.name}; paper DOI: {mcp._chem_repro_paper_doi}; dataset: {dataset_scope}.\n"
        + (tool.description or "")
        + f"\nReturns ONE complete AnalysisResult JSON object. Suggested direct output name: {name}.json; "
          f"bind that output to this exact tool name ({name}). Downloadable CSVs and figures are listed "
          "inside artifacts; their actual filenames are returned by execution, not guessed in advance. "
          "For a comparison mode, all labeled variants belong to this one JSON result."
    )
    if tool.output_schema is not None:
        tool.output_schema["description"] = (
            f"Complete scientific result of {name}. Suggested direct output name: {name}.json. "
            "Preserve the returned answer, metadata and artifact inventory as one object."
        )
    tool.parameters["x-data-contract"] = {
        "input_mode": input_mode, "dataset_scope": dataset_scope,
        # Ordinary Python signature defaults are operational: explicitly passing the default
        # has the same effect as omitting that optional argument. Clients may canonicalize
        # redundant defaults only against an identical, already approved schema snapshot.
        "optional_numeric_defaults_are_operational": (
            input_mode == "fixed_dataset" and getattr(mcp, "_chem_repro_operational_numeric_defaults", False)
        ),
    }
    mcp.add_tool(tool)


def serve(mcp: FastMCP, *, default_port: int):
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--host", default=os.getenv("CHEM_REPRO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CHEM_REPRO_PORT", str(default_port))))
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        mcp.run(transport="http", host=args.host, port=args.port, path="/mcp", show_banner=False)
