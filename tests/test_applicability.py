import numpy as np
import pytest
from rdkit import DataStructs

from chem_repro.common.applicability import ApplicabilityDomain, nearest_distances


def bits(values, size=2048):
    fp = DataStructs.ExplicitBitVect(size)
    for value in values:
        fp.SetBit(value)
    return fp


def test_dense_fingerprints_do_not_overflow_uint8():
    fp = bits(range(600))
    assert nearest_distances([fp], [fp] * 6, k=5)[0] == 0


def test_self_exclusion_matches_independent_set_jaccard():
    sets = [{1, 2}, {1}, {2}, {3}, {1, 2, 3}, {4}, {1, 4}]
    fps = [bits(s) for s in sets]
    expected = np.array([
        np.mean(sorted(1 - len(s & t) / len(s | t) for j, t in enumerate(sets) if i != j)[:3])
        for i, s in enumerate(sets)
    ])
    actual = nearest_distances(fps, fps, k=3, exclude_indices=list(range(len(fps))))
    np.testing.assert_allclose(actual, expected)
    domain = ApplicabilityDomain.fit(fps, k=3, sample=7)
    np.testing.assert_allclose(domain.reference_distances, expected[domain.reference_indices])


def test_degenerate_reference_and_insufficient_training_data():
    fp, other = bits({1}), bits({2})
    domain = ApplicabilityDomain.fit([fp] * 6)
    assert domain.threshold == 0
    np.testing.assert_array_equal(domain.score([fp, other])["reference_domain_score_percent"], [100, 0])
    with pytest.raises(ValueError, match="Insufficient"):
        ApplicabilityDomain.fit([fp] * 5)
