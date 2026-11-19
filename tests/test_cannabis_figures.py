import pandas as pd
import pytest
from chem_repro.cannabis.analysis import result
from chem_repro.cannabis.figures import complete
from chem_repro.common.datasets import table
from chem_repro.common.artifacts import export_tables,artifact_root


def test_plot_reader_recovers_complete_hashed_table_not_preview(tmp_path,monkeypatch):
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    rows=pd.DataFrame({'name':['endpoint_'+str(i) for i in range(200)],'value':[i/200 for i in range(200)]})
    out=export_tables(result('all rows',[],{'values':table(rows,'complete')},'test'),port=7341)
    assert out.answer.tables['values'].is_preview
    recovered=complete(out,'values')
    pd.testing.assert_frame_equal(recovered,rows)
    digest=out.answer.tables['values'].full_table_artifact_sha256
    (artifact_root()/(digest+'.csv')).write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'):complete(out,'values')
