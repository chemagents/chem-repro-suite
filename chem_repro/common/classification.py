"""Binary evaluation with explicit operating points and separate probability/margin scores."""

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def binary_metrics(y, scores, *, threshold, score_kind="probability"):
    y, s = np.asarray(y), np.asarray(scores, float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Require aligned finite scores and binary observations with both classes")
    if score_kind not in {"probability", "decision_margin"} or not np.isfinite(threshold):
        raise ValueError("Declare the score type and a finite operating threshold")
    if score_kind == "probability" and ((s < 0) | (s > 1)).any():
        raise ValueError("Probabilities must lie in [0,1]")
    if score_kind == "probability" and not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must lie in [0,1]")
    prediction = s >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, s)
    return {
        "n_test": len(y),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "threshold": float(threshold),
        "score_kind": score_kind,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fp / (fp + tn)),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "predicted_positive_fraction": float(prediction.mean()),
        "accuracy": float((tp + tn) / len(y)),
        "brier_score": float(brier_score_loss(y, s)) if score_kind == "probability" else None,
    }


def evaluate_product(y, structure_probabilities, rte_probabilities, *, threshold=0.5):
    """Evaluate already aligned scores. Product is a ranking rule, not calibrated joint probability."""
    a, b = np.asarray(structure_probabilities, float), np.asarray(rte_probabilities, float)
    result = {
        "structure": binary_metrics(y, a, threshold=threshold),
        "rte": binary_metrics(y, b, threshold=threshold),
    }
    result["product"] = binary_metrics(y, a * b, threshold=threshold)
    result["operating_point_effect"] = {
        "delta_fpr": result["product"]["fpr"] - result["structure"]["fpr"],
        "delta_recall": result["product"]["recall"] - result["structure"]["recall"],
        "structure_positive_retention": float(
            ((a >= threshold) & (a * b >= threshold)).sum() / (a >= threshold).sum()
        )
        if (a >= threshold).any()
        else None,
    }
    return result
