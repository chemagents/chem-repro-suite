import numpy as np
import pytest
from pydantic import ValidationError

from chem_repro.common.chemistry import (
    catalyst_identity,
    ee_to_ddg,
    morgan_fingerprints,
    scaffold,
    tanimoto_distances,
)
from chem_repro.common.contracts import Metric, Table


def test_axial_identity_is_not_collapsed():
    core = "c1ccc2ccccc2c1-c1ccc2ccccc2c1"
    assert catalyst_identity(core, "R") != catalyst_identity(core, "S")
    with pytest.raises(ValueError):
        catalyst_identity(core, "")


def test_point_chirality_and_salts_are_preserved():
    assert catalyst_identity("N[C@H](C)C(=O)O") != catalyst_identity("N[C@@H](C)C(=O)O")
    assert "." in catalyst_identity("C[NH3+].[Cl-]")[0]


def test_ee_energy_and_units():
    assert ee_to_ddg(0, 25) == 0
    assert ee_to_ddg(2, 25) == pytest.approx(0.0237001223534528, abs=1e-14)
    assert ee_to_ddg(100, 25) == ee_to_ddg(99.9, 25)
    assert ee_to_ddg(90, 50) > ee_to_ddg(90, 25)
    for e, t in [(-1, 25), (101, 25), (50, -273.15), (float("nan"), 25)]:
        with pytest.raises(ValueError):
            ee_to_ddg(e, t)


def test_acyclic_scaffold_not_replaced_with_molecule():
    assert scaffold("CCCC") == ""
    assert scaffold("Cc1ccccc1") == "c1ccccc1"


def test_tanimoto_identity_and_symmetry():
    fps = morgan_fingerprints(["CCO", "CCO", "c1ccccc1"], chirality=True)
    d = tanimoto_distances(fps)
    assert np.allclose(d, d.T)
    assert d[0, 1] == 0 and d[0, 2] > 0


def test_contract_forbids_nonfinite_and_missing_table_fields():
    with pytest.raises(ValidationError):
        Metric(name="r2", value=float("nan"), population="test")
    with pytest.raises(ValidationError):
        Table(description="x", columns=["a", "b"], rows=[{"a": 1}])


def test_redocking_rejects_inconsistent_ranks_and_keeps_failures_in_denominator():
    import pandas as pd

    from chem_repro.common.docking import pose_recovery_summary

    frame = pd.DataFrame(
        [
            {
                "target": "a",
                "evaluated": True,
                "best_top5_rmsd_A": 1.0,
                "first_recovery_rank": 2,
                "exclusion_reason": None,
            },
            {
                "target": "b",
                "evaluated": True,
                "best_top5_rmsd_A": 3.0,
                "first_recovery_rank": None,
                "exclusion_reason": None,
            },
            {
                "target": "c",
                "evaluated": False,
                "best_top5_rmsd_A": None,
                "first_recovery_rank": None,
                "exclusion_reason": "apo",
            },
        ]
    )
    recovery, failed, excluded, median = pose_recovery_summary(frame, rmsd_threshold=2.0)
    assert median == 2.0
    assert recovery.iloc[-1].fraction_recovered == 0.5
    assert list(failed.target) == ["b"] and list(excluded.target) == ["c"]
    frame.loc[1, "first_recovery_rank"] = 1
    with pytest.raises(ValueError, match="conflict"):
        pose_recovery_summary(frame, rmsd_threshold=2.0)


def test_provenance_refuses_hot_modified_source(monkeypatch):
    from chem_repro.common import datasets

    assert len(datasets.implementation_sha256()) == 64
    monkeypatch.setattr(datasets, "_source_sha256", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="restart"):
        datasets.implementation_sha256()


def test_table_preview_preserves_complete_checksummed_download(tmp_path,monkeypatch):
    import hashlib

    import pandas as pd

    from chem_repro.common.artifacts import export_tables
    from chem_repro.common.contracts import AnalysisResult, Answer, Provenance
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    original=Table(description='all observations',columns=['id'],rows=[{'id':i} for i in range(65)])
    output=AnalysisResult(status='success',answer=Answer(finding='test',tables={'full':original}),
                          metadata=Provenance(study_id='test',paper_doi='test',protocol_id='test',sources=[],software={}))
    output=export_tables(output,port=7342)
    preview=output.answer.tables['full']
    assert len(preview.rows)==20 and preview.total_rows==65 and preview.is_preview
    artifact=output.artifacts[0]
    assert artifact.sha256==preview.full_table_artifact_sha256
    path=tmp_path/(artifact.sha256+'.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact.sha256
    assert pd.read_csv(path).id.tolist()==list(range(65))
    assert len(original.rows)==65
    # Export is idempotent for an already previewed table; it never exports the preview as full data.
    assert len(export_tables(output,port=7342).artifacts)==1


async def test_shared_registration_exports_unhandled_tables_for_every_adapter(tmp_path,monkeypatch):
    import pandas as pd
    from fastmcp import Client

    from chem_repro.common.contracts import AnalysisResult, Answer, Provenance
    from chem_repro.common.pool import register, study_server
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    server=study_server('AutoExportTest','test',artifact_port=12345)
    def analyze() -> AnalysisResult:
        return AnalysisResult(status='success',answer=Answer(finding='computed',tables={
            'records':Table(description='all rows',columns=['id'],rows=[{'id':i} for i in range(80)])}),
            metadata=Provenance(study_id='test',paper_doi='test',protocol_id='test',sources=[],software={}))
    register(server,analyze,name='analyze',dataset_scope='test')
    async with Client(server) as client:
        response=await client.call_tool('analyze',{})
        data=response.structured_content
        assert data['answer']['tables']['records']['total_rows']==80
        assert len(data['answer']['tables']['records']['rows'])==20
        assert data['answer']['tables']['records']['is_preview']
        artifact=data['artifacts'][0]
        assert pd.read_csv(tmp_path/(artifact['sha256']+'.csv')).id.tolist()==list(range(80))


def test_compact_summary_retains_roles_after_the_first_twenty_rows(tmp_path, monkeypatch):
    from chem_repro.common.artifacts import export_tables
    from chem_repro.mannich.analysis import component_classes
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR', str(tmp_path))
    result = export_tables(component_classes(), port=7343)
    frequencies = result.answer.tables['class_frequencies']
    assert len(frequencies.rows) == frequencies.total_rows == 40
    assert not frequencies.is_preview
    assert {row['role'] for row in frequencies.rows} == {'organocatalyst', 'nucleophile', 'electrophile'}
    assert len(result.answer.tables['class_representatives'].rows) == 11


def test_wide_rows_do_not_escape_inline_budget_and_remain_in_full_csv(tmp_path, monkeypatch):
    import pandas as pd

    from chem_repro.common.artifacts import export_tables
    from chem_repro.common.contracts import AnalysisResult, Answer, Provenance
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR', str(tmp_path))
    rows = [{'id': 1, 'text': 'x' * 20000}]
    result = AnalysisResult(status='success', answer=Answer(finding='computed', tables={
        'wide': Table(description='wide row', columns=['id', 'text'], rows=rows)}),
        metadata=Provenance(study_id='test', paper_doi='test', protocol_id='test', sources=[], software={}))
    result = export_tables(result, port=7343)
    data = result.answer.tables['wide']
    assert data.rows == [] and data.total_rows == 1 and data.is_preview
    saved = pd.read_csv(tmp_path / (data.full_table_artifact_sha256 + '.csv'))
    assert saved.text.tolist() == ['x' * 20000]


def test_repeated_export_of_small_tables_is_idempotent(tmp_path,monkeypatch):
    from chem_repro.common.artifacts import export_tables
    from chem_repro.common.contracts import AnalysisResult, Answer, Provenance
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    output=AnalysisResult(status='success',answer=Answer(finding='test',tables={'small':Table(description='one',columns=['x'],rows=[{'x':1}])}),metadata=Provenance(study_id='test',paper_doi='test',protocol_id='test',sources=[],software={}))
    export_tables(output,port=12345)
    export_tables(output,port=12345)
    assert len(output.artifacts)==1


def test_failed_artifact_write_leaves_no_partial_file(tmp_path,monkeypatch):
    import errno

    from chem_repro.common import artifacts
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    original=artifacts.tempfile.NamedTemporaryFile
    class FailingWrite:
        def __init__(self,**kwargs):
            self.file=original(**kwargs)
            self.name=self.file.name
        def __enter__(self):
            self.file.__enter__();return self
        def write(self,payload):
            self.file.write(payload[:1])
            raise OSError(errno.ENOSPC,'simulated disk full')
        def __exit__(self,*args):return self.file.__exit__(*args)
    monkeypatch.setattr(artifacts.tempfile,'NamedTemporaryFile',FailingWrite)
    with pytest.raises(OSError):
        artifacts.save_artifact(b'complete result',suffix='json',media_type='application/json',public_base_url='http://localhost',display_name='result')
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("opt_in", [False, True])
async def test_operational_default_guarantee_requires_explicit_producer_opt_in(opt_in):
    from fastmcp import Client

    from chem_repro.common.pool import register, study_server

    server = study_server("DefaultGuarantee", "test", operational_numeric_defaults=opt_in)
    def analyze(threshold: float = .65) -> dict:
        return {"threshold": threshold}
    register(server, analyze, name="analyze", dataset_scope="test")
    async with Client(server) as client:
        tool = (await client.list_tools())[0]
        assert tool.inputSchema["x-data-contract"]["optional_numeric_defaults_are_operational"] is opt_in
        omitted = await client.call_tool("analyze", {})
        explicit = await client.call_tool("analyze", {"threshold": .65})
        assert omitted.structured_content == explicit.structured_content
