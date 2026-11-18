"""Reload every exported model and reproduce its saved held-out scores from raw residue matrices."""

import argparse
import io
import json
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from chem_repro.cannabis.data import residue_matrix
from chem_repro.common.artifacts import artifact_root


def artifact_bytes(record):
    path = artifact_root() / Path(urlparse(record["download_url"]).path).name
    data = path.read_bytes()
    assert sha256(data).hexdigest() == record["sha256"] and len(data) == record["size_bytes"]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    for path in sorted(args.results.glob("split_*.json")):
        result = json.loads(path.read_text())
        payload = json.loads(
            artifact_bytes(next(a for a in result["artifacts"] if a["name"].endswith("predictions.json")))
        )
        archive = artifact_bytes(next(a for a in result["artifacts"] if a["name"].endswith("models.zip")))
        predictions = pd.DataFrame(payload["predictions"])
        with (
            zipfile.ZipFile(io.BytesIO(archive)) as models,
            tempfile.TemporaryDirectory(prefix="chem-repro-reload-") as td,
        ):
            for (index, target, model_name), group in predictions.groupby(["split_index", "target", "model"]):
                prefix = f"split_{index:02d}/{target}/{model_name}"
                features = json.loads(models.read(prefix + "/features.json"))
                x = np.column_stack(
                    [
                        residue_matrix(t).loc[group.ligand_id, name].to_numpy(float)
                        for t, name in (f.split(":", 1) for f in features)
                    ]
                )
                x = np.where(np.isnan(x), 0, x)
                assert np.isfinite(x).all()
                if model_name == "CatBoost":
                    model_path = Path(td) / "model.json"
                    model_path.write_bytes(models.read(prefix + "/model.json"))
                    model = CatBoostClassifier()
                    model.load_model(str(model_path), format="json")
                    scores = model.predict_proba(x, thread_count=2)[:, 1]
                else:
                    params = json.loads(models.read(prefix + "/linear_parameters.json"))
                    assert params["classes"] == [0, 1]
                    scores = (
                        ((x - np.array(params["mean"])) / np.array(params["scale"]))
                        @ np.array(params["coef"]).T
                        + np.array(params["intercept"])
                    ).ravel()
                delta = float(np.max(np.abs(scores - group.score.to_numpy(float))))
                assert delta < 1e-10, (index, target, model_name, delta)
                checks.append(
                    {
                        "split_index": int(index),
                        "target": target,
                        "model": model_name,
                        "n_predictions": len(group),
                        "max_absolute_roundtrip_error": delta,
                    }
                )
        print(path.name, "model score roundtrips verified", flush=True)
    receipt = {
        "kind": "exported_models_reloaded_against_raw_inputs",
        "models_verified": len(checks),
        "predictions_verified": sum(c["n_predictions"] for c in checks),
        "max_absolute_error": max(c["max_absolute_roundtrip_error"] for c in checks),
        "checks": checks,
    }
    (args.results / "model_roundtrip_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print({k: v for k, v in receipt.items() if k != "checks"}, flush=True)


if __name__ == "__main__":
    main()
