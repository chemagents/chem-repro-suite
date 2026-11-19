from chem_repro.common.completed_cache import CompletedResultCache
from chem_repro.common.artifacts import save_artifact
from chem_repro.common.contracts import AnalysisResult, Answer, Provenance


def test_completed_call_is_reused_explicitly_but_missing_files_and_force_recompute_are_not(monkeypatch,tmp_path):
    monkeypatch.setenv("CHEM_REPRO_ARTIFACTS_DIR",str(tmp_path))
    calls=[]
    def calculate():
        calls.append(1)
        artifact=save_artifact(b"original complete bytes",suffix="zip",media_type="application/zip",public_base_url="http://127.0.0.1:7343",display_name="structures.zip")
        return AnalysisResult(status="partial",answer=Answer(finding="computed, unresolved stereochemistry retained"),
            metadata=Provenance(study_id="study",paper_doi="10.1/test",protocol_id="generation",sources=[],software={}),artifacts=[artifact])
    cache=CompletedResultCache()
    first=cache.call("generate",{"seed":42},calculate)
    second=cache.call("generate",{"seed":42},calculate)
    assert len(calls)==1 and not first.metadata.reused_from_cache and second.metadata.reused_from_cache
    assert second.status=="partial" and second.artifacts==first.artifacts
    second.answer.finding="edited by caller"
    assert "edited" not in cache.call("generate",{"seed":42},calculate).answer.finding
    cache.call("generate",{"seed":42},calculate,force_recompute=True)
    assert len(calls)==2
    (tmp_path/(first.artifacts[0].sha256+".zip")).unlink()
    fresh=cache.call("generate",{"seed":42},calculate)
    assert len(calls)==3 and not fresh.metadata.reused_from_cache
    cache.call("generate",{"seed":43},calculate)
    assert len(calls)==4
