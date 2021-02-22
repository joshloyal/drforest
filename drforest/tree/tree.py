import numbers
import six

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._tree import dimension_reduction_tree


__all__ = ['DimensionReductionTreeRegressor']


class DimensionReductionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_slices=10,
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=2,
                 sdr_algorithm=None,
                 random_state=123):
        self.n_slices = n_slices
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_save = min_samples_save
        self.sdr_algorithm = sdr_algorithm
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape

        # check input arrays
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # check parameters
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = n_features
            else:
                raise ValueError("Unrecognized value for max_features")
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:
            max_features = int(self.max_features * n_features)

        if not (0 < max_features <= n_features):
            raise ValueError("max_features must be in (0, n_features]")

        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got {}".format(
                                    self.min_samples_leaf))
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got {}".format(
                                    self.min_samples_leaf))
            min_samples_leaf = int(np.ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_save, (numbers.Integral, np.integer)):
            if not 1 <= self.min_samples_save:
                raise ValueError("min_samples_save must be at least 1 "
                                 "or in (0, 0.5], got {}".format(
                                    self.min_samples_save))
            min_samples_save = self.min_samples_save
        else:  # float
            if not 0.0 < self.min_samples_save <= 1.0:
                raise ValueError("min_samples_save must be at least 1 "
                                 "or in (0, 1.0], got {}".format(
                                    self.min_samples_save))
            min_samples_save = int(np.ceil(self.min_samples_save * n_samples))

        if isinstance(self.sdr_algorithm, six.string_types):
            if self.sdr_algorithm not in ["sir", "save"]:
                raise ValueError("sdr_algorithm must be one of "
                                 "{'sir', 'save'}. got {}".format(
                                    self.sdr_algorithm))
            sdr_algorithm = 0 if self.sdr_algorithm == 'sir' else 1

        # set sample_weight
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        self.tree_ = dimension_reduction_tree(
            X, y, sample_weight,
            num_slices=self.n_slices,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            seed=self.random_state)

        # fit overall SDR direction using slices determined by the leaf nodes
        if self.sdr_algorithm is not None:
            self.directions_ = self.tree_.estimate_sufficient_dimensions(
                    X, sdr_algorithm)

        return self

    def predict(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.predict(X)

    def transform(self, X):
        check_is_fitted(self, 'directions_')

        return np.dot(X, self.directions_.T)

    def apply(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.apply(X)

    def decision_path(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.decision_path(X)
