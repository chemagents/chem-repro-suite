"""One scientific comparison can cover the complete model-by-endpoint grid."""

import pandas as pd

from ..common.datasets import table
from .analysis import metric, result, transfer
from .data import ENDPOINTS


def compare_transfer(endpoint="both", model="both", identity_policy="strict", bootstrap=1000,
                     include_oof=True, include_figures=True):
    if endpoint not in {"both", "neurotoxicity", "eye_corrosion"}:
        raise ValueError("Unsupported endpoint")
    if model not in {"both", "RF", "ElasticNet"}:
        raise ValueError("Unsupported model")
    endpoints = ["neurotoxicity", "eye_corrosion"] if endpoint == "both" else [endpoint]
    models = ["RF", "ElasticNet"] if model == "both" else [model]
    completed = {(name, target): transfer(target, name, identity_policy, bootstrap, include_oof)
                 for name in models for target in endpoints}
    if len(completed) == 1:
        output = next(iter(completed.values()))
        if include_figures:
            from .figures import transfer as plot
            output = plot(output)
        return output
    merged = {}
    penalties, descriptions, records = [], [], []
    for (name, target), part in completed.items():
        for key, tab in part.answer.tables.items():
            merged.setdefault(key, []).extend({"model": name, "endpoint": target, **row} for row in tab.rows)
        measures = {m.name:m.value for m in part.answer.metrics}
        metrics = {row["scheme"]:row for row in part.answer.tables["validation_metrics"].rows}
        if set(metrics) != {"random", "scaffold"}:
            raise ValueError("Every model/endpoint must complete both split schemes")
        records.append(measures["n_records"])
        penalties.append({"model": name, "endpoint": target, "delta_r2": measures["delta_r2"]})
        descriptions.append(f"{name}/{target}: R² {metrics['random']['r2']:.3f} → {metrics['scaffold']['r2']:.3f}")
    if len(set(records)) != 1:
        raise ValueError("Grid comparisons require the same source population")
    tables = {key: table(pd.DataFrame(rows), {
        "validation_metrics":"All model × endpoint × split combinations; pooled OOF R²/RMSE/MAE and bootstrap intervals",
        "fold_audit":"Every fold with its model/endpoint; reported overlap refers to the declared generic scaffold groups",
        "oof_predictions":"All held-out molecule predictions; model/endpoint/scheme/fold and exact generic-scaffold groups retained for independent overlap checks",
    }[key]) for key, rows in merged.items()}
    tables["transfer_penalties"] = table(pd.DataFrame(penalties), "Random minus scaffold pooled OOF R²")
    # Preserve the child protocols' distinct qualifications when combining their results.
    base_codes = {"MODEL_DERIVED_ENDPOINTS", "SOURCE_IDENTITY_CONFLICT"}
    notes = list({(note.code, note.message, note.severity): note
                  for part in completed.values() for note in part.answer.limitations
                  if note.code not in base_codes}.values())
    output = result(
        f"Completed all {len(completed)} requested model/endpoint combinations, each with random and scaffold "
        "five-fold validation. " + "; ".join(descriptions) + ". Targets are ADMETlab model outputs, not measured toxicity.",
        [metric("model_endpoint_combinations", len(completed)),
         metric("split_evaluations", len(completed) * 2), metric("n_records_per_combination", records[0])],
        tables, protocol="heracleum.surrogate_grid.v1", notes=notes,
        models=models, endpoints=endpoints, identity_policy=identity_policy, bootstrap=bootstrap,
        target_columns={target: ENDPOINTS[target] for target in endpoints},
        seed=42, morgan_radius=2, morgan_bits=2048,
        child_protocols=[{"model": name, "endpoint": target, "protocol": part.metadata.protocol_id,
                          "parameters": part.metadata.protocol_parameters} for (name, target), part in completed.items()],
    )
    output.metadata.reused_from_cache = all(part.metadata.reused_from_cache for part in completed.values())
    if include_figures:
        if len(completed) == 4:
            from .figures import transfer_comparison
            output.artifacts.extend(transfer_comparison(completed))
        else:
            from .figures import transfer as plot
            for part in completed.values():
                output.artifacts.extend(plot(part).artifacts)
    return output
