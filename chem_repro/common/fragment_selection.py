"""Train-only feature selection with an explicit fragment-count denominator.

This implements the documented frequency and positive-Pearson filters. A supplied
dictionary and its feature order remain part of the protocol; this class does not
reconstruct the unpublished Syntelly dictionary or atom-location correlation rule.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FragmentSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fragment_columns, presence_min=0.01, count_per_molecule_min=0.025,
                 descriptor_nonzero_min=0.025, positive_correlation_max=0.85):
        self.fragment_columns = fragment_columns
        self.presence_min = presence_min
        self.count_per_molecule_min = count_per_molecule_min
        self.descriptor_nonzero_min = descriptor_nonzero_min
        self.positive_correlation_max = positive_correlation_max

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or X.empty or X.columns.duplicated().any():
            raise ValueError("Require a nonempty frame with unique named features")
        if not np.isfinite(X.to_numpy(float)).all():
            raise ValueError("Nonfinite features require explicit handling, not zero imputation")
        if not set(self.fragment_columns) <= set(X):
            raise ValueError("Fragment columns are absent")
        if not 0 <= self.presence_min <= 1 or not 0 <= self.descriptor_nonzero_min <= 1:
            raise ValueError("Occurrence thresholds must be fractions")
        if self.count_per_molecule_min < 0 or not 0 <= self.positive_correlation_max <= 1:
            raise ValueError("Invalid count or correlation threshold")
        fragments = set(self.fragment_columns)
        if (X[list(fragments)].to_numpy(float) < 0).any():
            raise ValueError("Fragment counts cannot be negative")
        kept, fragment_kept, audit = [], [], []
        for name in X:
            values = X[name].to_numpy(float)
            frequency = float(np.mean(values != 0))
            count_mean = float(values.mean()) if name in fragments else None
            reason, correlated_with, correlation = "retained", None, None
            if name in fragments:
                if frequency < self.presence_min:
                    reason = "fragment_presence_below_threshold"
                elif count_mean < self.count_per_molecule_min:
                    reason = "fragment_counts_per_training_molecule_below_threshold"
                elif np.std(values) > 0:
                    for previous in fragment_kept:
                        other = X[previous].to_numpy(float)
                        if np.std(other) == 0:
                            continue
                        coefficient = float(np.corrcoef(values, other)[0, 1])
                        if coefficient > self.positive_correlation_max:
                            reason, correlated_with, correlation = "positive_fragment_correlation", previous, coefficient
                            break
            elif frequency < self.descriptor_nonzero_min:
                reason = "descriptor_nonzero_below_threshold"
            if reason == "retained":
                kept.append(name)
                if name in fragments:
                    fragment_kept.append(name)
            audit.append({"feature": name, "kind": "fragment" if name in fragments else "descriptor",
                          "nonzero_fraction": frequency, "count_per_training_molecule": count_mean,
                          "constant_on_training": bool(np.std(values) == 0), "decision": reason,
                          "correlated_with": correlated_with, "pearson_r": correlation})
        if not kept:
            raise ValueError("No features survive the declared training-only selection")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.selected_features_ = kept
        self.selection_audit_ = pd.DataFrame(audit)
        self.n_training_records_ = len(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction features must have exactly the fitted names and order")
        if not np.isfinite(X.to_numpy(float)).all():
            raise ValueError("Nonfinite prediction features are not silently replaced")
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
