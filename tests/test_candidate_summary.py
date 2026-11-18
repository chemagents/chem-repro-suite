import pandas as pd
import pytest

from chem_repro.cannabis.prioritization import summarize_mapped_references


def test_subset_summary_does_not_infer_rank_range_from_paper_order():
    frame = pd.DataFrame({
        "paper_name": ["first in paper", "highest source score", "last in paper"],
        "source_rank_population_size": [100, 100, 100],
        "source_product_rank_before_annotations": [20, 7, 35],
        "computed_product_score": [.4, .8, .1],
        "source_structure_screen_positive": [True, True, False],
    })
    metrics, finding = summarize_mapped_references(frame, source_population=100, scheme="random")
    values = {m.name: m.value for m in metrics}
    assert values["mapped_reference_product_rank_min"] == 7
    assert values["mapped_reference_product_rank_max"] == 35
    assert values["mapped_reference_screen_positive"] == 2
    assert values["mapped_reference_screen_negative"] == 1
    assert "highest source score" in finding and "7–35" in finding
    assert all("not a selected source top10" in m.population for m in metrics)
    with pytest.raises(ValueError, match="population"):
        summarize_mapped_references(frame, source_population=99, scheme="random")
