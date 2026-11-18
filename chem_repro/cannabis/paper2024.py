"""Documented 2024 preparation, substitute descriptors and declared search spaces.

These building blocks do not modify the frozen v1 predictions. Source: Molecules
29(8),1826, section4.3 and Supplement S1. Missing historical choices are explicit.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.model_selection import ParameterGrid

from ..common.training_records import prepare_training_records

ALLOWED_ELEMENTS = ("N", "I", "As", "O", "B", "Br", "F", "P", "Se", "S", "C", "Cl", "Si")
ASSUMPTIONS = {
    "identity": "RDKit canonical isomeric SMILES; isotopes/charges/fragments retained; atom maps removed",
    "hydrogen": "Explicit isotopic hydrogen is allowed; the published whitelist describes heavy elements",
    "sources": "Collection labels are not recovered constituent experimental sources",
    "fragment_dictionary": "RDKit fr_* substitute, not the unpublished author custom dictionary",
    "count_threshold": "sum(fragment counts)/training molecules >=0.025; denominator not explicit in paper",
    "correlation": "Greedy feature-order positive Pearson r>0.85 on training columns; atom-location rule unavailable",
    "folds": "Selection must be refitted inside every inner/outer training partition",
    "search": "Original Optuna trial count, sampler/seed and exact CV assignments are unspecified",
    "catboost_bounds": "Published interval endpoints treated as inclusive",
    "xgboost_reduced": "Use reported depth12, learning_rate0.01,2000trees; unspecified other values are library defaults",
}


def prepare(frame, task, source_dataset, regression_aggregation="record_mean"):
    if "source_dataset" not in frame:
        frame = frame.assign(source_dataset=source_dataset)
    return prepare_training_records(frame, task=task, allowed_elements=ALLOWED_ELEMENTS,
                                    regression_aggregation=regression_aggregation)


def descriptor_frame(smiles):
    """Return explicit substitute features; fail on undefined descriptors."""
    descriptors = dict(Descriptors._descList)
    fragments = [name for name in descriptors if name.startswith("fr_")]
    names = fragments + ["element_" + symbol for symbol in ALLOWED_ELEMENTS]
    names += ["HeavyAtomCount", "bonds_single", "bonds_double", "bonds_triple", "bonds_aromatic",
              "NumValenceElectrons", "NumRotatableBonds", "TPSA", "MolLogP", "LabuteASA", "Kappa1", "Kappa2", "Kappa3"]
    names += [name for name in descriptors if name.startswith(("SlogP_VSA", "SMR_VSA", "EState_VSA"))]
    records = []
    bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for raw in smiles:
        mol = Chem.MolFromSmiles(raw)
        if mol is None or not mol.GetNumAtoms():
            raise ValueError("Descriptor calculation requires valid prepared molecules")
        row = {name: float(descriptors[name](mol)) for name in names if name in descriptors}
        row.update({"element_" + symbol: sum(a.GetSymbol() == symbol for a in mol.GetAtoms())
                    for symbol in ALLOWED_ELEMENTS})
        row.update({"bonds_" + label: sum(b.GetBondType() == kind for b in mol.GetBonds())
                    for label, kind in zip(["single", "double", "triple", "aromatic"], bonds)})
        row["NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
        records.append(row)
    frame = pd.DataFrame(records, columns=names, dtype=float)
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Empty or nonfinite descriptor matrix; no implicit zero imputation")
    return frame, fragments


def catboost_search_parameters(trial):
    """Optuna-compatible conditional space; do not send incompatible bootstrap args."""
    p = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
         "iterations": trial.suggest_int("iterations", 100, 1000),
         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
         "depth": trial.suggest_int("depth", 3, 11),
         "random_strength": trial.suggest_float("random_strength", 1e-7, 20, log=True),
         "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
         "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])}
    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    return p


def xgboost_search_space(*, full):
    """Full S1 grid or its four-choice reduced grid, without launching any fits."""
    if full:
        return ParameterGrid({"max_depth": list(range(1, 14)), "learning_rate": [0.1, 0.01, 0.001],
                              "subsample": [1, 0.75, 0.5, 0.3], "n_estimators": [1500, 2000, 2500, 3000],
                              "reg_lambda": [1, 2, 4], "reg_alpha": [0, 10, 40],
                              "colsample_bytree": [1, 0.75, 0.5, 0.3], "min_child_weight": [1, 5, 10]})
    return ParameterGrid({"max_depth": [12], "learning_rate": [0.01], "n_estimators": [2000],
                          "subsample": [1, 0.75, 0.5, 0.3]})
