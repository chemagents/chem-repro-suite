"""Recompute reaction PCA with declared choices where the supplement is underspecified."""

import io
import json
import os
import zipfile
from functools import lru_cache
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ..common.artifacts import save_artifact
from ..common.contracts import EvidenceNote
from ..common.datasets import table
from ..common.reactions import structural_reaction_fingerprint
from .analysis import _classified, _metric, _result


@lru_cache(maxsize=2)
def coordinates(include_agents=True):
    df = _classified().copy()
    matrix = np.asarray(
        [
            np.asarray(
                structural_reaction_fingerprint(
                    [r.nucleophile, r.electrophile],
                    [r.product],
                    [r.organocatalyst],
                    include_agents=include_agents,
                )
            )
            for r in df.itertuples()
        ],
        dtype=np.uint8,
    )
    with threadpool_limits(limits=2):
        estimator = PCA(n_components=2, svd_solver="arpack", tol=1e-10, random_state=42)
        embedded = estimator.fit_transform(matrix)
    frame = df[
        [
            "reaction_id",
            "nucleophile",
            "electrophile",
            "organocatalyst",
            "axial_configuration",
            "product",
            "normalized_doi",
            "organocatalyst_computed_class",
        ]
    ].copy()
    frame["PC1"], frame["PC2"] = embedded[:, 0], embedded[:, 1]
    frame["fingerprint_sha256"] = [sha256(np.packbits(row).tobytes()).hexdigest() for row in matrix]
    loadings = pd.DataFrame(estimator.components_.T, columns=["PC1", "PC2"])
    loadings.insert(0, "bit_index", np.arange(matrix.shape[1]))
    return frame, estimator.explained_variance_ratio_, loadings, np.packbits(matrix, axis=1)


def reaction_space(include_agents=True):
    before = coordinates.cache_info().hits
    frame, fractions, loadings, fingerprints = coordinates(include_agents)
    result = _result(
        f"Projected all {len(frame)} curated reaction records using an explicitly declared structural "
        f"reaction fingerprint; PC1/PC2 explain {fractions[0]:.1%}/{fractions[1]:.1%}. "
        "The supplement names RDKit structural fingerprints but omits their settings and PCA scaling. "
        "This computation is a documented reconstruction, not a verified match to the original PCA coordinates.",
        [
            _metric("reaction_records", len(frame)),
            _metric("unique_structural_fingerprints", frame.fingerprint_sha256.nunique()),
            _metric("fingerprint_bits", 4096),
        ],
        {
            "explained_variance": table(
                pd.DataFrame({"component": ["PC1", "PC2"], "fraction": fractions}),
                "Centered, unscaled binary structural reaction fingerprints",
            ),
            "coordinate_preview": table(
                frame.head(20), "First 20 rows only; complete 4015-row coordinates in the artifact"
            ),
        },
        protocol="mannich.reaction_pca.declared_rdkit_structural.v1",
        status="partial",
        fingerprint="CreateStructuralFingerprintForReaction / PatternFP",
        fp_size=4096,
        include_agents=include_agents,
        agent_bit_ratio=0.2,
        non_agent_weight=1,
        agent_weight=1,
        role_mapping={
            "reactants": ["nucleophile", "electrophile"],
            "agent": "organocatalyst",
            "product": "product",
        },
        centered=True,
        standardized=False,
        pca_solver="arpack",
        pca_tolerance=1e-10,
        seed=42,
        rdkit_default_source="https://github.com/rdkit/rdkit/blob/Release_2026_03_3/Code/GraphMol/ChemReactions/ReactionFingerprints.cpp",
        notes=[
            EvidenceNote(
                code="PUBLISHED_FIGURE_NOT_REPRODUCED",
                message="Visual inspection of the "
                "original supplementary PCA plot shows a different coordinate scale and point-cloud "
                "shape from this reconstruction. Bounded L2-normalization/agent-block diagnostics did "
                "not resolve the mismatch. The generated PCA must not be described as a reproduced "
                "original Figure S4; exact source settings or coordinates remain required.",
            ),
            EvidenceNote(
                code="REACTION_PCA_SETTINGS_UNREPORTED",
                message="The deposited supplement does "
                "not specify fingerprint type/length, agent inclusion, scaling or solver. This run makes "
                "those choices explicit, using the installed structural-function defaults and unscaled PCA. "
                "The original Figure S4 numerical coordinates/loadings are unavailable for exact comparison.",
            ),
            EvidenceNote(
                code="REACTION_REPRESENTATION_SCOPE",
                message="The fingerprint is not an atom-mapped "
                "reaction center. It omits conditions and ee/yield, and does not encode the separate axial "
                "R/S annotation. Catalysts occupy the agent block when enabled, never the reactant block. "
                "Identical fingerprints can therefore represent distinct experimental records.",
            ),
        ],
    )
    result.metadata.reused_from_cache = coordinates.cache_info().hits > before
    stream = io.BytesIO()
    packed = io.BytesIO()
    np.savez_compressed(
        packed, packed_bits=fingerprints, bit_order=np.array("big"), fingerprint_length=np.array(4096)
    )
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("coordinates.csv", frame.to_csv(index=False))
        archive.writestr("bit_loadings.csv", loadings.to_csv(index=False))
        archive.writestr("fingerprints.npz", packed.getvalue())
        archive.writestr("protocol.json", json.dumps(result.metadata.model_dump(mode="json"), indent=2))
    result.artifacts.append(
        save_artifact(
            stream.getvalue(),
            suffix="zip",
            media_type="application/zip",
            public_base_url=os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", "http://127.0.0.1:7343"),
            display_name=f"mannich_reaction_pca_agents_{include_agents}.zip",
        )
    )
    return result
