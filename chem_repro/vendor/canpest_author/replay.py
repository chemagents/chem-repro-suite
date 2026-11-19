"""Execute unchanged canpest functions, adapting only paths and missing orchestration.

This reproduces the deposited RMT95 branch, not the later RTE390 geometric mix.
The original selection intentionally uses all labels and in-sample inner PR-AUC.
Those source limitations are preserved and are not called independent validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import threading
import shutil

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "replay_config.json").read_text())
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["OMP_NUM_THREADS"] = str(CONFIG["torch_threads"])
os.environ["OPENBLAS_NUM_THREADS"] = str(CONFIG["torch_threads"])
sys.path.insert(0, str(ROOT / "source_code"))
import numpy as np
import pandas as pd
import torch
import joblib
import rdkit
import sklearn
import catboost
import train_models as train
import select_rmt_residues as select
import run_split_veto as one
import run_30splits_veto as many


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(model_dir, data, device):
    """Independently load exported weights/scaler and reproduce every test score."""
    saved = pd.read_csv(model_dir / "test_predictions.csv")
    metrics = json.loads((model_dir / "metrics.json").read_text())
    scaler = joblib.load(model_dir / "feature_scaler.pkl")
    names = np.load(model_dir / "global_feature_names.npy", allow_pickle=True).tolist()
    assert names == train.rdkit2d_names()
    full = train.load_rdkit2d(data.SMILES.to_numpy())
    mapping = {int(v): i for i, v in enumerate(data.ligand_id)}
    loc = [mapping[int(v)] for v in saved.ligand_id]
    smiles = data.iloc[loc].SMILES.to_numpy()
    features = scaler.transform(full[loc])
    model = train.DMPNNClassifier(features.shape[1], CONFIG["hidden_size"], CONFIG["depth"],
                                  CONFIG["dropout"], CONFIG["ffn_hidden"], CONFIG["ffn_layers"]).to(device)
    model.load_state_dict(torch.load(model_dir / "dmpnn_model.pt", map_location=device, weights_only=True))
    predictions = train.predict_proba(model, smiles, features, np.arange(len(saved)), device)
    error = float(np.max(np.abs(predictions - saved.dmpnn_proba)))
    # CSV float32 serialization and different batch grouping can cause float32 roundoff.
    if not np.allclose(predictions, saved.dmpnn_proba, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Reloaded source checkpoint differs: {error}")
    assert np.array_equal(data.iloc[loc].activity.astype(int), saved.activity.astype(int))
    return {"n_test": len(saved), "max_reload_abs_error": error,
            "model_sha256": sha(model_dir / "dmpnn_model.pt"),
            "predictions_sha256": sha(model_dir / "test_predictions.csv"),
            "blend_weight_dmpnn": metrics["blend_weight_dmpnn"]}


def main():
    manifest = json.loads((ROOT / "author_source_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Author input changed: {name}")
    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    request = {"config": CONFIG, "source_manifest_sha256": sha(ROOT / "author_source_manifest.json"),
               "runner_sha256": sha(__file__), "torch": torch.__version__, "rdkit": rdkit.__version__,
               "numpy": np.__version__, "sklearn": sklearn.__version__, "catboost": catboost.__version__}
    old = out / "request.json"
    if old.exists() and json.loads(old.read_text()) != request:
        raise ValueError("Output directory belongs to a different immutable run")
    old.write_text(json.dumps(request, indent=2))
    print(json.dumps({"stage": "start", **request}), flush=True)
    torch.set_num_threads(CONFIG["torch_threads"])
    train.RDKIT_CACHE = out / "full_rdkit2d.npy"
    one.OUT_DIR = out
    one.RMT_FEATURE_DIR = out / "data/rmt95_features"
    one.MODEL_DIR = out / "models"
    one.DMPNN_DIR = out / "models/dmpnn_rdkit2d"
    one.PRED_DIR = out / "predictions"
    one.METRIC_DIR = out / "metrics"
    one.REPORT_DIR = out / "reports"
    one.SELECTION_CSV = out / "rmt_selection/selected_rmt95_features.csv"
    select.OUT_DIR = out / "rmt_selection"
    one.ensure_dirs()
    if not one.SELECTION_CSV.exists():
        select.main()
    selected = pd.read_csv(one.SELECTION_CSV)
    reference = pd.read_csv(ROOT / "rmt_selection/selected_rmt95_features.csv")
    selection_match = selected[["gene", "residue"]].equals(reference[["gene", "residue"]])
    print(json.dumps({"stage": "selection", "n_features": len(selected),
                      "ordered_membership_matches_source": selection_match}), flush=True)
    features = one.load_selected_feature_frames()
    data = pd.read_csv(one.DATA_CSV)
    data.ligand_id = data.ligand_id.astype(int)
    registry = pd.read_csv(one.SPLIT_CSV)
    assert data.ligand_id.is_unique
    device = torch.device(CONFIG["device"])
    args = argparse.Namespace(**CONFIG)
    all_metrics, all_predictions, checks = [], [], []
    for split in CONFIG["split_ids"]:
        t0 = time.time()
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(30):
                print(json.dumps({"stage": "running", "split": split,
                                  "elapsed_s": round(time.time()-t0)}), flush=True)
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            original_train, original_test = one.write_split_feature_csv(split, features, registry, data)
            for src, subset in [(original_train, "train"), (original_test, "test")]:
                shutil.copyfile(src, src.parent / f"rmt_rte_sel_{subset}.csv")
            model_dir = one.DMPNN_DIR / split
            complete = model_dir / "verified.json"
            if not complete.exists():
                train.train_dmpnn("rmt_rte_sel", split, data, one.RMT_FEATURE_DIR,
                                  out / "training", args, device, feature_set="rdkit2d")
                generated = out / "training/rdkit2d" / split
                model_dir.parent.mkdir(parents=True, exist_ok=True)
                if model_dir.exists():
                    raise ValueError("Incomplete model destination already exists")
                generated.rename(model_dir)
                checked = verify_checkpoint(model_dir, data, device)
                complete.write_text(json.dumps(checked, indent=2))
            else:
                checked = verify_checkpoint(model_dir, data, device)
            val = pd.read_csv(model_dir / "val_predictions.csv")
            test = pd.read_csv(model_dir / "test_predictions.csv")
            reg = registry[registry.split_id == split]
            train_ids = set(reg.loc[reg['set']=='train', 'ligand_id'])
            test_ids = set(reg.loc[reg['set']=='test', 'ligand_id'])
            assert set(test.ligand_id) == test_ids
            assert set(val.ligand_id) <= train_ids
            assert not train_ids & test_ids
            checked.update({"split_id": split, "source_test_ids_match": True,
                            "n_validation": len(val), "test_train_overlap": 0})
            measured, predictions, _ = many.process_split(split, features, registry, data, CONFIG['threshold'])
            all_metrics.append(measured)
            all_predictions.append(predictions)
            checks.append(checked)
            pd.concat(all_metrics).to_csv(out / "per_split_metrics.csv", index=False)
            pd.concat(all_predictions).to_csv(out / "heldout_predictions.csv", index=False)
            many.summarize(pd.concat(all_metrics)).to_csv(out / "summary.csv", index=False)
            (out / "verification.json").write_text(json.dumps(checks, indent=2))
            print(json.dumps({"stage":"split_complete","split":split,"seconds":round(time.time()-t0,1),
                              "metrics":measured.to_dict('records')}), flush=True)
        finally:
            stop.set()
            worker.join(timeout=1)
    receipt = {"complete":True,"splits":len(checks),"heldout_prediction_rows":len(pd.concat(all_predictions)),
               "ordered_selection_matches_source":selection_match,
               "protocol":"deposited_RMT95_code_retraining", "source_commit":manifest['commit'],
               "source_limitations":["global-label RMT selection and in-sample inner PR-AUC preserved",
                                      "not the RTE390 geometric mix", "new fits, not recovered historical weights"],
               "summary":many.summarize(pd.concat(all_metrics)).to_dict('records')}
    (out / "receipt.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({"stage":"complete",**receipt}),flush=True)


if __name__ == '__main__':
    main()
