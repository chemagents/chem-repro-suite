import numpy as np
import pytest

from chem_repro.common.classification import binary_metrics, evaluate_product


def test_product_reduction_has_recall_cost():
    results = evaluate_product([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], [0.9, 0.2, 0.2, 0.8])
    assert results["structure"]["fpr"] == 0.5
    assert results["product"]["fpr"] == 0
    assert results["operating_point_effect"]["delta_recall"] == -0.5
    assert results["product"]["tp"] == 1 and results["product"]["fn"] == 1


def test_margin_is_not_probability():
    values = [-3, -0.1, 0.2, 4]
    result = binary_metrics([0, 0, 1, 1], values, threshold=0, score_kind="decision_margin")
    assert result["roc_auc"] == 1 and result["brier_score"] is None
    with pytest.raises(ValueError, match="Probabilities"):
        binary_metrics([0, 0, 1, 1], values, threshold=0.5)
    with pytest.raises(ValueError, match="Probabilities"):
        evaluate_product([0, 0, 1, 1], [0.1, 0.1, 0.8, 0.9], values)


def test_empty_precision_is_not_silently_perfect():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.1, 0.2, 0.3], threshold=0.5)
    assert result["precision"] is None and result["recall"] == 0
    for values in [[0.1, 0.2], [0.1, 0.2, 0.3, np.nan]]:
        with pytest.raises(ValueError):
            binary_metrics([0, 0, 1, 1], values, threshold=0.5)
