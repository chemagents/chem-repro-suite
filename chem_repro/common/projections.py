"""Chemical-space projection with explicit fingerprint/distance settings."""

import numpy as np
from sklearn.manifold import TSNE

from .chemistry import morgan_fingerprints, tanimoto_distances


def fingerprint_tsne(smiles, *, chirality, distance_mode, seed, perplexity=30, max_iter=1000):
    if not 0 < perplexity < len(smiles):
        raise ValueError("Perplexity must be positive and smaller than the molecular population")
    fps = morgan_fingerprints(smiles, chirality=chirality)
    if distance_mode == "precomputed_tanimoto":
        features, metric = tanimoto_distances(fps), "precomputed"
    elif distance_mode == "jaccard_bits":
        features, metric = np.asarray([np.asarray(fp) for fp in fps], dtype=bool), "jaccard"
    else:
        raise ValueError("Unknown chemical distance mode")
    model = TSNE(
        n_components=2,
        metric=metric,
        init="random",
        perplexity=perplexity,
        random_state=seed,
        max_iter=max_iter,
        learning_rate="auto",
        n_jobs=1,
    )
    coordinates = model.fit_transform(features)
    return coordinates, float(model.kl_divergence_), len({fp.ToBitString() for fp in fps})
