"""Actual per-target and combined RTE model fits with fold-specific residue selection."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..common.artifacts import save_artifact, save_json
from ..common.classification import binary_metrics
from ..common.contracts import EvidenceNote
from ..common.datasets import table
from ..common.rmt import fit_residue_selector
from .analysis import metric, result
from .data import TARGETS, residue_matrix, split


def validate_rte(split_indices=(0,), *, iterations=500, include_models=False, progress=None):
    if (
        not split_indices
        or len(set(split_indices)) != len(split_indices)
        or any(not isinstance(i, int) or not 0 <= i < 30 for i in split_indices)
    ):
        raise ValueError("Require unique source split indices in 0..29")
    if not isinstance(iterations, int) or not 10 <= iterations <= 1000:
        raise ValueError("Use 10..1000 explicitly declared CatBoost iterations")
    scores, predictions, selection_rows, selection_details, audit_rows = [], [], [], [], []
    model_archive = io.BytesIO()
    with zipfile.ZipFile(model_archive, "w", compression=zipfile.ZIP_DEFLATED) as models:
        for index in split_indices:
            train, test = split(index)
            y_train, y_test = train.activity.to_numpy(int), test.activity.to_numpy(int)
            split_record = {
                "train_ids": train.ligand_id.astype(int).tolist(),
                "test_ids": test.ligand_id.astype(int).tolist(),
                "train_labels": y_train.tolist(),
                "test_labels": y_test.tolist(),
            }
            split_hash = hashlib.sha256(json.dumps(split_record, sort_keys=True).encode()).hexdigest()
            audit_rows.append(
                {
                    "split_index": index,
                    "split_sha256": split_hash,
                    "n_train": len(train),
                    "n_test": len(test),
                    "ligand_id_overlap": len(set(train.ligand_id) & set(test.ligand_id)),
                }
            )
            matrices = {}
            for target in TARGETS:
                matrix = residue_matrix(target)
                # .loc fails on absent ligand IDs rather than dropping unmatched rows.
                x_train = matrix.loc[train.ligand_id].to_numpy(float)
                x_test = matrix.loc[test.ligand_id].to_numpy(float)
                selector = fit_residue_selector(
                    x_train, y_train, list(matrix), inner_splits=5, inner_test_size=0.2, seed=42
                )
                selected_names = [selector.feature_names[i] for i in selector.selected_indices]
                matrices[target] = (
                    selector.transform(x_train, list(matrix)),
                    selector.transform(x_test, list(matrix)),
                    [target + ":" + name for name in selected_names],
                )
                selection_rows.append(
                    {
                        "split_index": index,
                        "target": target,
                        **selector.diagnostics,
                        "selected_features": selected_names,
                    }
                )
                selection_details.append(
                    {
                        "split_index": index,
                        "target": target,
                        "split_registry": split_record,
                        "ranked_residues": table(selector.ranking, "Train-only rank/contribution").rows,
                        "inner_scan": table(selector.scan, "Diagnostic inner ROC-AUC scan").rows,
                    }
                )
            matrices["ALL6"] = (
                np.column_stack([v[0] for v in matrices.values()]),
                np.column_stack([v[1] for v in matrices.values()]),
                [name for v in matrices.values() for name in v[2]],
            )
            for target, (x_train, x_test, names) in matrices.items():
                estimators = {
                    "CatBoost": CatBoostClassifier(
                        iterations=iterations,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        random_seed=42,
                        thread_count=2,
                        verbose=False,
                        allow_writing_files=False,
                        use_best_model=False,
                        task_type="CPU",
                    ),
                    "LinearSVM": make_pipeline(
                        StandardScaler(),
                        LinearSVC(
                            C=1.0, class_weight="balanced", dual="auto", max_iter=20000, random_state=42
                        ),
                    ),
                }
                for model, estimator in estimators.items():
                    estimator.fit(x_train, y_train)
                    if model == "CatBoost":
                        values = estimator.predict_proba(x_test)[:, 1]
                        operating_point, score_kind = 0.5, "probability"
                    else:
                        values = estimator.decision_function(x_test)
                        operating_point, score_kind = 0.0, "decision_margin"
                    row = {
                        "split_index": index,
                        "target": target,
                        "model": model,
                        "n_selected_features": len(names),
                        **binary_metrics(y_test, values, threshold=operating_point, score_kind=score_kind),
                    }
                    scores.append(row)
                    predictions.extend(
                        {
                            "split_index": index,
                            "target": target,
                            "model": model,
                            "ligand_id": int(lid),
                            "label": int(label),
                            "score": float(score),
                            "score_kind": score_kind,
                        }
                        for lid, label, score in zip(test.ligand_id, y_test, values)
                    )
                    if progress:
                        progress(row)
                    if include_models:
                        prefix = f"split_{index:02d}/{target}/{model}"
                        models.writestr(prefix + "/features.json", json.dumps(names))
                        if model == "CatBoost":
                            with tempfile.TemporaryDirectory(prefix="chem-repro-model-") as td:
                                path = Path(td) / "model.json"
                                estimator.save_model(str(path), format="json")
                                models.writestr(prefix + "/model.json", path.read_bytes())
                        else:
                            scaler, svc = estimator.steps[0][1], estimator.steps[1][1]
                            models.writestr(
                                prefix + "/linear_parameters.json",
                                json.dumps(
                                    {
                                        "mean": scaler.mean_.tolist(),
                                        "scale": scaler.scale_.tolist(),
                                        "coef": svc.coef_.tolist(),
                                        "intercept": svc.intercept_.tolist(),
                                        "classes": svc.classes_.tolist(),
                                        "prediction_formula": "((raw_energies-mean)/scale) @ coef.T + intercept",
                                    }
                                ),
                            )
    frame = pd.DataFrame(scores)
    summary = (
        frame.groupby(["target", "model"])
        .agg(
            n_splits=("split_index", "nunique"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_fpr=("fpr", "mean"),
            std_fpr=("fpr", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
        )
        .reset_index()
    )
    output = result(
        f"Executed {len(scores)} RTE fits across {len(split_indices)} source outer splits: CatBoost and "
        "linear SVM for each of six targets and their concatenation. Selection was refitted on each "
        "outer training set. Repeated holdout observations are not pooled as independent OOF records.",
        [metric("completed_model_fits", len(scores)), metric("completed_outer_splits", len(split_indices))],
        {
            "heldout_metrics": table(
                frame, "Actual test scores; PR area definitions and SVM margin threshold explicit"
            ),
            "split_summary": table(
                summary, "Arithmetic mean and sample SD across source outer splits; SD null for one split"
            ),
            "residue_selections": table(
                pd.DataFrame(selection_rows), "A distinct Appendix A selection for each target and split"
            ),
            "split_audit": table(pd.DataFrame(audit_rows), "Exact split identities and hashes"),
        },
        "cannabis.rte_validation.appendix_a.v1",
        notes=[
            EvidenceNote(
                code="FINAL_RUN_CONFIG_UNAVAILABLE",
                message="The paper does not specify exact "
                "CatBoost/SVM hyperparameters, final fold identities or all inner-selection settings. "
                "This run uses the explicit implementation settings in metadata and the public source "
                "split registry. It is a new method reproduction; no exact final-paper metric match is claimed.",
            ),
            EvidenceNote(
                code="SVM_SCORE_TYPE",
                message="Linear SVM returns uncalibrated decision margins; "
                "its native threshold is zero. CatBoost is evaluated at the paper's probability threshold "
                "0.5. SVM margins are never multiplied as probabilities in the fusion rule.",
            ),
            EvidenceNote(
                code="RESAMPLING_SCOPE",
                message="The public 30 splits are repeated stratified "
                "holdouts. Test observations repeat across splits; their SD is descriptive variability, "
                "not an independent-sample standard error. Test rows are not used for selection or stopping.",
            ),
        ],
        split_indices=list(split_indices),
        rmt_seed=42,
        inner_splits=5,
        inner_test_size=0.2,
        candidate_m="1..p",
        contribution_fraction=0.98,
        catboost={
            "iterations": iterations,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "random_seed": 42,
            "thread_count": 2,
            "task_type": "CPU",
            "early_stopping": False,
        },
        svm={
            "implementation": "LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 20000,
            "random_state": 42,
            "scaling": "StandardScaler fitted to outer training only",
        },
        pr_auc_definitions="average precision and trapezoidal PR-AUC both reported",
        outer_test_used_in_fit=False,
        fold_aggregation="unweighted mean and sample SD",
    )
    output.metadata.software["catboost"] = version("catboost")
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7341")
    output.artifacts.append(
        save_json(
            {
                "metadata": output.metadata.model_dump(mode="json"),
                "predictions": predictions,
                "selections_and_splits": selection_details,
            },
            public_base_url=base,
            display_name="cannabis_rte_validation_predictions.json",
        )
    )
    if include_models:
        output.artifacts.append(
            save_artifact(
                model_archive.getvalue(),
                suffix="zip",
                media_type="application/zip",
                public_base_url=base,
                display_name="cannabis_rte_fitted_models.zip",
            )
        )
    return output
