"""CSV transport must not silently convert integer values through float64."""
import csv
import json

from chem_repro.common.artifacts import save_table
from chem_repro.common.contracts import Table


def test_nullable_integer_columns_preserve_exact_identifiers_and_tiers(tmp_path, monkeypatch):
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR', str(tmp_path))
    table = Table(description='nullable identifiers and ranks',
                  columns=['id', 'tier', 'score', 'label', 'details'], rows=[
                      {'id': 9007199254740993, 'tier': 1, 'score': 0.125,
                       'label': 'quoted, "label"\nnext line', 'details': {'a': [1, None]}},
                      {'id': None, 'tier': None, 'score': None, 'label': '', 'details': None}])
    artifact = save_table(table, port=7342, display_name='nullable.csv')
    with (tmp_path / (artifact.sha256 + '.csv')).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['id'] == '9007199254740993'
    assert rows[0]['tier'] == '1'
    assert rows[0]['score'] == '0.125'
    assert rows[0]['label'] == table.rows[0]['label']
    assert json.loads(rows[0]['details']) == table.rows[0]['details']
    assert all(value == '' for value in rows[1].values())


def test_unchanged_tier_has_same_csv_spelling_with_or_without_unranked_rows(tmp_path, monkeypatch):
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR', str(tmp_path))
    exports = []
    for tiers in [[1, 2, 3], [1, None, None]]:
        table = Table(description='candidate tiers', columns=['tier'],
                      rows=[{'tier': value} for value in tiers])
        artifact = save_table(table, port=7342, display_name='tiers.csv')
        with (tmp_path / (artifact.sha256 + '.csv')).open(newline='') as handle:
            exports.append(list(csv.DictReader(handle)))
    assert exports[0][0]['tier'] == exports[1][0]['tier'] == '1'
    assert exports[1][1]['tier'] == ''
