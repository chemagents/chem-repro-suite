"""Training-reference Tanimoto kNN domain; the score is not a calibrated probability.

Method parameters follow chemagents/heracleum-tox-mcp-server (k=5, sigma=2,
500 deterministic training references). RDKit bit-vector arithmetic avoids uint8
dot-product overflow. Self-exclusion uses the actual row index before selection.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


def fingerprints(molecules):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if any(mol is None or mol.GetNumAtoms() == 0 for mol in molecules):
        raise ValueError("Applicability requires valid nonempty molecular graphs")
    return [generator.GetFingerprint(mol) for mol in molecules]


def nearest_distances(query, train, k=5, exclude_indices=None):
    if k < 1 or len(train) < k + int(exclude_indices is not None):
        raise ValueError("Insufficient reference molecules for requested k")
    if exclude_indices is not None and len(exclude_indices) != len(query):
        raise ValueError("Each training-reference query must identify its own row")
    out = []
    for i, fp in enumerate(query):
        distances = 1 - np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train), dtype=float)
        if exclude_indices is not None:
            index = exclude_indices[i]
            if not 0 <= index < len(train):
                raise ValueError("Invalid self-exclusion index")
            distances[index] = np.inf
        out.append(np.partition(distances, k - 1)[:k].mean())
    return np.asarray(out)


@dataclass
class ApplicabilityDomain:
    train: list
    threshold: float
    reference_indices: np.ndarray
    reference_distances: np.ndarray
    k: int = 5
    sigma: float = 2.0

    @classmethod
    def fit(cls, train, *, k=5, sigma=2.0, sample=500):
        if not np.isfinite(sigma) or sigma < 0 or sample < 1:
            raise ValueError("Invalid domain parameters")
        indices = np.random.default_rng(0).choice(len(train), size=min(sample, len(train)), replace=False)
        distances = nearest_distances([train[i] for i in indices], train, k, indices)
        threshold = float(distances.mean() + sigma * distances.std(ddof=0))
        return cls(train, threshold, indices, distances, k, sigma)

    def score(self, query):
        distances = nearest_distances(query, self.train, self.k)
        if self.threshold == 0:
            percentages = np.where(distances == 0, 100.0, 0.0)
        else:
            percentages = 100 * np.exp(-np.square(distances / self.threshold))
        return {
            "mean_knn_distance": distances,
            "reference_domain_score_percent": percentages,
            "outside_reference_distance": distances > self.threshold,
        }
