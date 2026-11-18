import asyncio
import threading

import pytest
from fastmcp import Client

from chem_repro.common.pool import register, study_server


async def test_discovery_remains_responsive_during_blocking_scientific_work():
    mcp = study_server("ResponsiveTest", "test")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(value: int = 7) -> dict[str, int]:
        started.set()
        release.wait(timeout=5)
        finished.set()
        return {"value": value}

    register(mcp, blocked, name="blocked", dataset_scope="test")
    async with Client(mcp) as worker, Client(mcp) as discovery:
        task = asyncio.create_task(worker.call_tool("blocked", {"value": 9}))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            tools = await discovery.list_tools()
            assert not finished.is_set(), "Scientific work blocked metadata discovery"
            assert tools[0].inputSchema["properties"]["value"]["type"] == "integer"
            assert tools[0].outputSchema["type"] == "object"
        finally:
            release.set()
        response = await task
        assert response.structured_content == {"value": 9}


async def test_scientific_work_remains_serialized():
    mcp = study_server("SerialTest", "test")
    started, release = threading.Event(), threading.Event()
    active, peak = 0, 0

    def calculation(first: bool) -> dict[str, bool]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if first:
            started.set()
            release.wait(timeout=5)
        active -= 1
        return {"first": first}

    register(mcp, calculation, name="calculate", dataset_scope="test")
    async with Client(mcp) as one, Client(mcp) as two:
        first = asyncio.create_task(one.call_tool("calculate", {"first": True}))
        assert await asyncio.to_thread(started.wait, 3)
        second = asyncio.create_task(two.call_tool("calculate", {"first": False}))
        try:
            await two.list_tools()
        finally:
            release.set()
        await asyncio.gather(first, second)
        assert peak == 1


async def test_cancelled_request_does_not_release_a_running_scientific_operation():
    mcp = study_server("CancellationTest", "test")
    started, release = threading.Event(), threading.Event()
    active, peak = 0, 0

    def calculation(first: bool) -> dict[str, bool]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if first:
            started.set()
            release.wait(timeout=5)
        active -= 1
        return {"first": first}

    register(mcp, calculation, name="calculate", dataset_scope="test")
    async with Client(mcp) as one, Client(mcp) as two:
        first = asyncio.create_task(one.call_tool("calculate", {"first": True}))
        assert await asyncio.to_thread(started.wait, 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(two.call_tool("calculate", {"first": False}))
        try:
            await two.list_tools()
        finally:
            release.set()
        await second
        assert peak == 1
