"""Figures 4 and 6 numerical projections; catalyst identities retain axial annotations."""

from functools import lru_cache
from hashlib import sha256
import json

import pandas as pd

from ..common.contracts import EvidenceNote
from ..common.datasets import DATA_ROOT, table
from ..common.projections import fingerprint_tsne
from .analysis import _classified, _metric, _result, load


def chemical_space(population="catalysts", protocol="notebook"):
    before = _chemical_space.cache_info().hits
    result = _chemical_space(population, protocol).model_copy(deep=True)
    result.metadata.reused_from_cache = _chemical_space.cache_info().hits > before
    return result


@lru_cache(maxsize=4)
def _chemical_space(population, protocol):
    if population not in {"catalysts", "reagents"} or protocol not in {"notebook", "chirality_sensitivity"}:
        raise ValueError("Unknown population or projection protocol")
    if population == "catalysts":
        df = _classified()
        counts = df.catalyst_identity.value_counts()
        frame = df.drop_duplicates("catalyst_identity")[
            [
                "organocatalyst_canonical",
                "axial_configuration",
                "organocatalyst_computed_class",
                "catalyst_identity",
            ]
        ].copy()
        frame["n_reactions"] = frame.catalyst_identity.map(counts)
        frame = frame.drop(columns="catalyst_identity").rename(
            columns={"organocatalyst_canonical": "smiles", "organocatalyst_computed_class": "class"}
        )
        mode, seed = "precomputed_tanimoto", 0
        chirality = protocol == "chirality_sensitivity"
    else:
        df = load()
        # Literal source deduplication in figure_06: first role wins; do not silently canonicalize it.
        seen = set()
        rows = []
        for role in ["nucleophile", "electrophile"]:
            for smiles in df[role].unique():
                if smiles not in seen:
                    seen.add(smiles)
                    rows.append(
                        {
                            "smiles": smiles,
                            "role": role,
                            "n_reactions_in_assigned_role": int((df[role] == smiles).sum()),
                            "occurs_in_both_roles": bool(
                                (df.nucleophile == smiles).any() and (df.electrophile == smiles).any()
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        mode, seed, chirality = "jaccard_bits", 42, True
    coords, kl, n_fps = fingerprint_tsne(
        frame.smiles.tolist(), chirality=chirality, distance_mode=mode, seed=seed
    )
    frame.insert(0, "projection_id", [f"{population}:{i:04d}" for i in range(1, len(frame) + 1)])
    # This digest covers every identity row and label, not t-SNE coordinates.
    identity_rows = table(frame, "Full pre-projection identity mapping").rows
    identity_digest = sha256(json.dumps(identity_rows, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    identity_audit = pd.DataFrame([{
        "population": population, "source_identity_rows": len(frame),
        "unique_projection_ids": frame.projection_id.nunique(),
        "identity_mapping_sha256": identity_digest,
        "identity_columns": ",".join(frame.columns),
        "audit_scope": "All input identity rows retained; digest excludes coordinates. This is a computation audit, not independent source curation.",
    }])
    evidence = json.loads((DATA_ROOT / "mannich/projection_method_evidence.json").read_text())
    frame["tsne_x"], frame["tsne_y"] = coords[:, 0], coords[:, 1]
    return _result(
        f"Projected {len(frame)} {population} using the declared Morgan/t-SNE protocol; "
        f"{n_fps} distinct fingerprints. Axial annotations remain in catalyst identities but are "
        "not encoded by ordinary Morgan bits. Projection geometry is descriptive, not activity evidence.",
        [
            _metric("projected_records", len(frame), population=population),
            _metric("distinct_fingerprints", n_fps, population=population),
            _metric("tsne_kl_divergence", kl, population=population),
        ],
        {"coordinates": table(frame, "Every source projection record, labels and numerical coordinates"),
         "identity_mapping_audit": table(identity_audit, "Complete identity mapping, excluding coordinates"),
         "catalyst_method_source_evidence": table(pd.DataFrame([evidence]), "Article/notebook comparison for catalyst Figure 4A", evidence_kind="source_record")},
        protocol=f"mannich.{population}_tsne.{protocol}.v2",
        morgan_radius=2,
        morgan_bits=2048,
        include_chirality=chirality,
        distance_mode=mode,
        seed=seed,
        perplexity=30,
        max_iter=1000,
        init="random",
        learning_rate="auto",
        notes=[
            EvidenceNote(
                code="PAPER_NOTEBOOK_CHIRALITY",
                message="Paper prose calls catalyst fingerprints "
                "chirality-sensitive; deposited figure_04 omits the chirality argument (default False) and explicitly comments NO chirality encoding. Notebook mode "
                "reproduces that code. chirality_sensitivity is a declared alternative for point "
                "stereochemistry; neither encodes the separate axial R/S column.",
            ),
            EvidenceNote(
                code="EMBEDDING_SCOPE",
                message="t-SNE axes, cluster orientation and global distances "
                "are not chemical observables. Exact coordinates can vary with library versions; the "
                "recorded parameters and source membership determine the reproducible analysis.",
            ),
        ],
    )
