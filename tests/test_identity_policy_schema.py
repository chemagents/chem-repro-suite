from fastmcp import Client
from chem_repro.heracleum.server import mcp

async def test_both_policy_tools_preserve_strict_default_and_explain_paper_replay():
    async with Client(mcp) as client:
        tools = {t.name:t for t in await client.list_tools()}
    for name in ["heracleum2026_scaffold_transfer", "heracleum2026_prioritize_candidates"]:
        prop = tools[name].inputSchema["properties"]["identity_policy"]
        assert prop["default"] == "strict"
        assert prop["enum"] == (["strict", "published_snapshot", "compare"] if name.endswith("prioritize_candidates") else ["strict", "published_snapshot"])
        assert "published_snapshot for numerical reproduction" in prop["description"]
        assert "Strict can change the analyzed population" in prop["description"]
