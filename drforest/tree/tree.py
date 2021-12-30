import numbers

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._tree import dimension_reduction_tree


__all__ = ['DimensionReductionTreeRegressor', 'DecisionTreeRegressor']


class DimensionReductionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_slices=10,
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=3,
                 categorical_cols=None,
                 sdr_algorithm=None,
                 random_state=123):
        self.n_slices = n_slices
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.categorical_cols = categorical_cols
        self.sdr_algorithm = sdr_algorithm
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape

        # check input arrays
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # check parameters
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = n_features
            else:
                raise ValueError("Unrecognized value for max_features")
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:
            max_features = max(int(self.max_features * n_features), 1)

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

        if isinstance(self.sdr_algorithm, str):
            if self.sdr_algorithm not in ["sir", "save"]:
                raise ValueError("sdr_algorithm must be one of "
                                 "{'sir', 'save'}. got {}".format(
                                    self.sdr_algorithm))
            sdr_algorithm = 0 if self.sdr_algorithm == 'sir' else 1

        # set sample_weight
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        if self.categorical_cols is not None:
            self.categorical_features_ = np.asarray(
                self.categorical_cols, dtype=int)
            self.numeric_features_ = np.asarray(
                [i for i in np.arange(n_features) if
                    i not in self.categorical_features_],
                dtype=int)
        else:
            self.categorical_features_ = np.asarray([], dtype=int)
            self.numeric_features_ = np.arange(n_features)

        self.tree_ = dimension_reduction_tree(
            X, y, sample_weight,
            self.numeric_features_,
            self.categorical_features_,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_original_features=False,
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


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 n_slices=10,
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=3,
                 random_state=123):
        self.n_slices = n_slices
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape

        # check input arrays
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)

        # check parameters
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = n_features
            else:
                raise ValueError("Unrecognized value for max_features")
        elif self.max_features is None:
            max_features = n_features
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:
            max_features = max(int(self.max_features * n_features), 1)

        if not (0 < max_features <= n_features):
            raise ValueError("max_features must be in (0, n_features], "
                             "but got max_features = {}".format(max_features))

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

        # set sample_weight
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        self.tree_ = dimension_reduction_tree(
            X, y, sample_weight,
            np.asarray([], dtype=int), np.asarray([], dtype=int),
            num_slices=self.n_slices,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            use_original_features=True,
            seed=self.random_state)

        return self

    def predict(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.predict(X)

    def apply(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.apply(X)

    def decision_path(self, X):
        check_is_fitted(self, 'tree_')

        return self.tree_.decision_path(X)
