"""CPU Chemprop reconstruction of Cannabis Table S1 with resumable completed folds.

Author folds/checkpoints and the final 3403-compound training cohort are unavailable.
All numerical choices outside Table S1 are declared. Saved source IDs, source
SMILES and per-fold train/validation/test membership remain available for audit.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from chem_repro.common.classification import binary_metrics
from chem_repro.common.dmpnn import (
    TrainOnlyDescriptorScaler,
    build_loader,
    create_model,
    descriptor_matrix,
    molecule_dataset,
    source_structure_view,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    body = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(body)
    temporary.replace(path)


class Curve(pl.Callback):
    def __init__(self, path):
        self.path, self.rows = path, []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = {"epoch": int(trainer.current_epoch),
               "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"])}
        for key, value in trainer.callback_metrics.items():
            if key in {"val_loss", "val/roc", "train_loss", "train_loss_epoch"}:
                row[key] = float(value.detach().cpu())
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print(json.dumps({"fold": self.path.parent.name, **row}), flush=True)


def predict(model, dataset, batch_size):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in build_loader(dataset, batch_size=batch_size):
            rows.append(model(batch[0], batch[1], batch[2]).cpu().numpy().reshape(-1))
    values = np.concatenate(rows)
    if len(values) != len(dataset) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must cover every input exactly once with finite probabilities")
    return values


def partitions(frame, args):
    y = frame.activity.to_numpy(int)
    if args.scheme == "source_random":
        registry = pd.read_csv(args.data_dir / "split_registry.csv")
        lookup = {value: i for i, value in enumerate(frame.ligand_id)}
        indices = [int(value) for value in args.split_indices.split(",")]
        if not indices or len(indices) != len(set(indices)) or any(not 0 <= i < 30 for i in indices):
            raise ValueError("Source split indices must be unique and within 0..29")
        outer = []
        for fold in indices:
            rows = registry.loc[registry.split_index == fold]
            if len(rows) != len(frame) or rows.ligand_id.duplicated().any() or set(rows.ligand_id) != set(lookup):
                raise ValueError("Source registry does not cover the complete labeled cohort")
            if any(y[lookup[r.ligand_id]] != r.activity for r in rows.itertuples()):
                raise ValueError("Source registry labels disagree with source data")
            outer.append((fold, np.array([lookup[i] for i in rows.loc[rows['set'] == 'train', 'ligand_id']]),
                          np.array([lookup[i] for i in rows.loc[rows['set'] == 'test', 'ligand_id']])))
    else:
        splitter = (StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                    if args.scheme == "random" else
                    StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed))
        outer = [(i, tr, te) for i, (tr, te) in enumerate(
            splitter.split(np.zeros(len(y)), y, frame.scaffold.to_numpy()))]
    for fold, outer_train, test in outer:
        if args.scheme == "scaffold":
            inner = StratifiedGroupKFold(10, shuffle=True, random_state=args.seed + fold)
            tr, va = next(inner.split(np.zeros(len(outer_train)), y[outer_train],
                                     frame.scaffold.to_numpy()[outer_train]))
            train, val = outer_train[tr], outer_train[va]
        else:
            train, val = train_test_split(outer_train, test_size=.1, stratify=y[outer_train],
                                         random_state=args.seed + fold)
        sets = [set(train), set(val), set(test)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)) or set.union(*sets) != set(range(len(y))):
            raise ValueError("Train/validation/test must form a disjoint complete partition")
        if any(set(y[index]) != {0, 1} for index in [train, val, test]):
            raise ValueError("One partition lacks a class; do not silently alter the split")
        cores = frame.scaffold.to_numpy()
        if args.scheme == "scaffold" and any(set(cores[a]) & set(cores[b])
                                              for a, b in [(train, val), (train, test), (val, test)]):
            raise ValueError("Scaffold leakage across training, validation or test")
        yield fold, train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheme", choices=["random", "scaffold", "source_random"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-indices", default="0")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 3 <= args.max_epochs <= 200 or not 2 <= args.folds <= 10 or not 1 <= args.threads <= 8:
        raise ValueError("Invalid explicitly bounded training configuration")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), code_root / "common/dmpnn.py", code_root / "common/classification.py"]
    source = args.data_dir / "data.csv"
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    for name in ["data.csv", "split_registry.csv"]:
        if digest(args.data_dir / name) != manifest["files"][name]["sha256"]:
            raise ValueError("Source bytes do not match the pinned article-source manifest")
    spec = {
        "protocol": "chemprop_paper_config_source_reconstruction_v1", "scheme": args.scheme,
        "source_sha256": digest(source), "split_registry_sha256": digest(args.data_dir / "split_registry.csv"),
        "source_revision": manifest["revision"],
        "code": {str(p.relative_to(code_root)): digest(p) for p in paths},
        "software": {p: version(p) for p in ["chemprop", "torch", "lightning", "numpy", "pandas", "rdkit", "scikit-learn", "scipy"]},
        "paper_configuration": {"depth": 3, "hidden": 300, "dropout": .1, "ffn_hidden": 300,
                                "ffn_linear_layers": 2, "aggregation": ["mean", "sum", "norm"],
                                "features": "normalized RDKit2D", "scheduler": "Noam", "output": "sigmoid"},
        "explicit_choices": {"norm": 100, "ffn_hidden_layers_in_chemprop_v2": 1,
                             "max_epochs": args.max_epochs, "patience": args.patience, "seed": args.seed,
                             "batch_size": args.batch_size, "threads": args.threads, "folds": args.folds,
                             "source_split_indices": args.split_indices if args.scheme == "source_random" else None,
                             "validation_fraction_random": .1, "inner_scaffold_folds": 10,
                             "scaffold_validation_choice": "first group fold; its row fraction varies with group sizes",
                             "validation_objective": "val/roc", "warmup_epochs": 2,
                             "learning_rates": [1e-4, 1e-3, 1e-4], "batch_norm": False,
                             "optimizer": "Adam", "gradient_clip_norm": 1.0, "class_balance_sampling": False,
                             "message_passing_bias": False, "atom_featurizer": "Chemprop v2 default (72 features)",
                             "bond_featurizer": "Chemprop default (14 features)",
                             "structure_view": "source connectivity with stereo explicitly removed; salts/charges retained and flagged",
                             "normalization": "train-only median imputation, mean/std, constant-feature removal, clip[-5,5]"},
        "fidelity": "New method reconstruction on the available docked source cohort; not an exact replay of author folds/checkpoints or final cohort",
    }
    spec_path = args.output / "run_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Output directory belongs to different inputs/code/software/settings")
    write_json(spec_path, spec)
    raw = pd.read_csv(source, dtype={"ligand_id": str})
    if set(raw.activity.dropna().unique()) != {0, 1}:
        raise ValueError("Source activity must contain only the two declared classes")
    raw["source_id_text"] = raw.ligand_id
    raw["ligand_id"] = raw.ligand_id.astype(int)
    if raw.ligand_id.duplicated().any():
        raise ValueError("Source ligand IDs are not unique")
    mols, audit = [], []
    for row in raw.itertuples():
        mol, record = source_structure_view(row.SMILES)
        mols.append(mol)
        audit.append({"ligand_id": row.ligand_id, "raw_smiles": row.SMILES, **record})
    audit = pd.DataFrame(audit)
    audit.to_csv(args.output / "structure_audit.csv", index=False)
    features, names = descriptor_matrix(mols)
    write_json(args.output / "descriptor_names.json", names)
    labeled = np.flatnonzero(raw.activity.notna().to_numpy())
    queried = np.flatnonzero(raw.activity.isna().to_numpy())
    frame = pd.concat([raw.iloc[labeled].reset_index(drop=True),
                       audit.iloc[labeled].drop(columns=["ligand_id", "raw_smiles"]).reset_index(drop=True)], axis=1)
    query_frame = raw.iloc[queried].reset_index(drop=True)
    query_mols = [mols[i] for i in queried]
    labeled_mols = [mols[i] for i in labeled]
    y = frame.activity.to_numpy(int)
    X, X_query = features[labeled], features[queried]
    full_predictions, full_queries, fit_summaries = [], [], []
    for fold, train, val, test in partitions(frame, args):
        fit_dir = args.output / f"fold_{fold:02d}"
        fit_dir.mkdir(exist_ok=True)
        identities = frame.analysis_smiles.to_numpy()
        cores = frame.scaffold.to_numpy()
        split = {"fold": fold, "train_indices": train.tolist(), "validation_indices": val.tolist(),
                 "test_indices": test.tolist(), "train_ids": frame.iloc[train].ligand_id.tolist(),
                 "validation_ids": frame.iloc[val].ligand_id.tolist(), "test_ids": frame.iloc[test].ligand_id.tolist(),
                 "identity_train_test_overlap": len(set(identities[train]) & set(identities[test])),
                 "scaffold_train_test_overlap": len(set(cores[train]) & set(cores[test])),
                 "scaffold_validation_test_overlap": len(set(cores[val]) & set(cores[test]))}
        done_path = fit_dir / "result.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done["split"] != split or done["run_spec_sha256"] != digest(spec_path):
                raise ValueError("Completed fold provenance differs")
            for name, expected in done["artifacts"].items():
                if digest(fit_dir / name) != expected:
                    raise ValueError("Completed fold artifact is corrupt")
            print(json.dumps({"fold": fold, "reused_completed_fold": True}), flush=True)
        else:
            pl.seed_everything(args.seed + fold, workers=True)
            scaler = TrainOnlyDescriptorScaler().fit(X[train])
            write_json(fit_dir / "scaler.json", scaler.to_dict())
            sets = [molecule_dataset([labeled_mols[i] for i in ix], y[ix], scaler.transform(X[ix]))
                    for ix in [train, val, test]]
            model = create_model(int(scaler.keep.sum()))
            ck = ModelCheckpoint(dirpath=fit_dir, filename="best", monitor="val/roc", mode="max", save_top_k=1)
            curve = Curve(fit_dir / "training_curve.csv")
            trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=args.max_epochs, logger=False,
                                 enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                                 callbacks=[curve, ck, EarlyStopping(monitor="val/roc", mode="max", patience=args.patience)],
                                 gradient_clip_val=1.0, deterministic="warn")
            trainer.fit(model, build_loader(sets[0], batch_size=args.batch_size, shuffle=True, seed=args.seed + fold),
                        build_loader(sets[1], batch_size=args.batch_size))
            if not ck.best_model_path:
                raise RuntimeError("Training produced no best validation checkpoint")
            # This checkpoint was just generated by this process, not supplied by a caller.
            saved = torch.load(ck.best_model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"])
            portable = fit_dir / "weights.pt"
            torch.save(model.state_dict(), portable)
            reloaded = create_model(int(scaler.keep.sum()))
            reloaded.load_state_dict(torch.load(portable, map_location="cpu", weights_only=True))
            p = predict(model, sets[2], args.batch_size)
            replay = predict(reloaded, sets[2], args.batch_size)
            if not np.array_equal(p, replay):
                raise ValueError("Portable checkpoint reload changed held-out probabilities")
            test_rows = frame.iloc[test][["ligand_id", "source_id_text", "activity", "SMILES", "analysis_smiles", "scaffold"]].copy()
            test_rows["probability"] = p
            test_rows["fold"] = fold
            test_rows.to_csv(fit_dir / "heldout_predictions.csv", index=False)
            q_data = molecule_dataset(query_mols, np.full(len(query_mols), np.nan), scaler.transform(X_query))
            query_rows = query_frame[["ligand_id", "source_id_text", "SMILES"]].copy()
            query_probabilities = predict(reloaded, q_data, args.batch_size)
            query_rows["probability"] = query_probabilities
            query_rows["fold"] = fold
            query_rows.to_csv(fit_dir / "query_predictions.csv", index=False)
            np.savez_compressed(fit_dir / "predictions.npz", test_ids=frame.iloc[test].ligand_id.to_numpy(),
                                y_test=y[test], probabilities=p,
                                query_ids=query_frame.ligand_id.to_numpy(), query_probabilities=query_probabilities)
            done = {"split": split, "run_spec_sha256": digest(spec_path), "best_epoch": int(saved["epoch"]),
                    "epochs_run": len(curve.rows), "best_validation_roc_auc": float(ck.best_model_score),
                    "n_descriptors_retained": int(scaler.keep.sum()), "checkpoint_reload_max_error": 0.0,
                    "metrics": binary_metrics(y[test], p, threshold=.5),
                    "artifacts": {path.name: digest(path) for path in fit_dir.iterdir() if path.is_file() and path.name != "result.json"}}
            write_json(done_path, done)
            print(json.dumps({"fold": fold, "completed": True, "metrics": done["metrics"]}), flush=True)
        fit_summaries.append({"fold": fold, "best_epoch": done["best_epoch"], "epochs_run": done["epochs_run"],
                              "best_validation_roc_auc": done["best_validation_roc_auc"], **done["metrics"]})
        full_predictions.append(pd.read_csv(fit_dir / "heldout_predictions.csv"))
        full_queries.append(pd.read_csv(fit_dir / "query_predictions.csv"))
    heldout, queries = pd.concat(full_predictions, ignore_index=True), pd.concat(full_queries, ignore_index=True)
    heldout.to_csv(args.output / "heldout_predictions.csv", index=False)
    queries.to_csv(args.output / "query_predictions_by_fold.csv", index=False)
    summary = pd.DataFrame(fit_summaries)
    summary.to_csv(args.output / "fold_metrics.csv", index=False)
    pooled = None
    if args.scheme != "source_random":
        if heldout.ligand_id.duplicated().any() or set(heldout.ligand_id) != set(frame.ligand_id):
            raise ValueError("OOF predictions must cover every labeled source record exactly once")
        pooled = binary_metrics(heldout.activity.to_numpy(int), heldout.probability.to_numpy(), threshold=.5)
    result = {"run_spec": spec, "n_labeled": len(frame), "n_queried": len(query_frame), "n_fits": len(fit_summaries),
              "pooled_oof_metrics": pooled, "fold_metrics": fit_summaries,
              "fold_mean_std": {name: {"mean": float(summary[name].mean()),
                                       "std": float(summary[name].std(ddof=1)) if len(summary) > 1 else None}
                                for name in ["roc_auc", "average_precision", "pr_auc_trapezoidal", "fpr", "recall"]},
              "source_diagnostics": {"source_stereo_removed_rows": int(audit.stereo_removed.sum()),
                                     "source_multifragment_rows": int((audit.source_fragments > 1).sum()),
                                     "source_charged_rows": int((audit.source_formal_charge != 0).sum()),
                                     "labeled_duplicate_structure_rows": int(frame.analysis_smiles.duplicated(keep=False).sum()),
                                     "labeled_crossclass_structure_groups": int((frame.groupby("analysis_smiles").activity.nunique() > 1).sum()),
                                     "query_training_structure_overlap_rows": int(audit.iloc[queried].analysis_smiles.isin(set(frame.analysis_smiles)).sum())},
              "top_level_artifacts": {p.name: digest(p) for p in args.output.iterdir() if p.is_file() and p.name != "result.json"}}
    if any(digest(p) != spec["code"][str(p.relative_to(code_root))] for p in paths):
        raise ValueError("Implementation changed during execution")
    write_json(args.output / "result.json", result)
    print(json.dumps({"completed": True, "scheme": args.scheme, "n_fits": len(fit_summaries), "pooled": pooled}), flush=True)


if __name__ == "__main__":
    main()
