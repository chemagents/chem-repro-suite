import pandas as pd
import pytest

from chem_repro.cannabis.prioritization import annotate_and_rank, prioritize_candidates


def test_candidate_annotation_identity_unknown_and_cutoffs():
    scores=pd.DataFrame({'ligand_id':[1,2,3], 'SMILES':['CCO','CCC','CCN'],
                         'structure_probability':[.8,.9,.69], 'rte_probability':[.5,.3,.99],
                         'product_score':[.4,.27,.6831]})
    unknown=annotate_and_rank(scores,[])
    assert not unknown.eligible.any() and set(unknown.annotation_status)=={'unknown'}
    annotation={'ligand_id':1,'source_smiles':'OCC','is_secondary_metabolite':True,
                'annotation_source':'curated test record'}
    ranked=annotate_and_rank(scores,[annotation]).set_index('ligand_id')
    # The0.5evaluation threshold does not remove an annotated0.70-positive candidate.
    assert ranked.loc[1,'eligible'] and ranked.loc[1,'product_score']<.5
    assert not ranked.loc[3,'structure_screen_positive']
    with pytest.raises(ValueError,match='SMILES disagrees'):
        annotate_and_rank(scores,[{**annotation,'source_smiles':'CCN'}])
    with pytest.raises(ValueError,match='Duplicate'):
        annotate_and_rank(scores,[annotation,annotation])


def test_published_candidate_scores_do_not_invent_component_probabilities(tmp_path,monkeypatch):
    monkeypatch.setenv('CHEM_REPRO_ARTIFACTS_DIR',str(tmp_path))
    output=prioritize_candidates(reference_only=True)
    rows=output.answer.tables['paper_top10'].rows
    assert len(rows)==10 and rows[0]['paper_final_score']==.997 and rows[-1]['paper_final_score']==.668
    assert all(row['paper_structure_probability'] is None and row['paper_rte_probability'] is None for row in rows)
    assert output.answer.tables['paper_top10'].evidence_kind=='paper_reported'
    assert output.status=='partial'
