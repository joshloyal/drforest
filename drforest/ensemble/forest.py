import numbers
import six

import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._forest import dimension_reduction_forest


__all__ = ['DimensionReductionForestRegressor']


def leaf_node_kernel(X_leaves, Y_leaves=None):
    return 1 - pairwise_distances(X_leaves, Y=Y_leaves, metric='hamming')


def local_direction(x0, X_train, weights, n_directions=1):
    # filter for data points with non-zero weights
    nonzero = weights != 0
    X_nonzero = X_train[nonzero] - x0

    w = weights[nonzero].reshape(-1, 1)
    X_nonzero -= (w * X_nonzero).sum(axis=0) / np.sum(w)
    M = np.dot(X_nonzero.T, X_nonzero * w) / np.sum(w)

    eigval, eigvec = np.linalg.eigh(M)

    return eigvec[:, :n_directions]


class DimensionReductionForestRegressor(BaseEstimator, RegressorMixin):
    """Dimension Reduction Forest Regressor.

    A dimension reduction forest is a random forest that uses
    sufficient dimension reduction to estimate the optimal linear
    combinations splits when building a regression tree.

    Currently the dimension reduction forest supports both
    Sliced Inverse Regression (SIR) and
    Sliced Average Variance Estimation (SAVE) as estimators for
    linear combination splits. The splitting criterion is reduction
    in mean squared error (variance) at each node.

    Parameters
    ----------
    n_estimators : int, optional (default=500)
        The number of trees in the forest.

    n_slices : int, optional (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded
        until leaves contain ``min_samples_leaf`` samples.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split.
        This is the amount of features going into the dimension reduction
        calculation at the node.

        - If int, then estimate SIR/SAVE using `max_features` features at each
          node.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are used to estimate
          SIR/SAVE at each node.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=int(log2(n_features + 1))`.
        - If None, then `max_features=n_features (same as "auto").

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be a leaf node. A split
        point at any depth will only be considered if it leaves at least
        ``min_samples_leaf`` training samples in each leaf and right branches.

    oob_mse : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the MSE of unseen data.

    store_X_y : bool, optional(default=False)
        Whether to store the training data X, y. This is required for
        calculating the random forest kernel.

    random_state : int, optional (default=42)
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the randomness inside the tree building process.

    n_jobs: int, optional(default=1)
        The number of jobs to run in parallel. ``fit``, ``predict`, and
        ``apply`` are all parallelized over the trees.
    """
    def __init__(self,
                 n_estimators=500,
                 n_slices=10,
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=3,
                 oob_mse=False,
                 store_X_y=False,
                 random_state=42,
                 n_jobs=1):
        self.n_estimators = n_estimators
        self.n_slices = n_slices
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.store_X_y = store_X_y
        self.n_jobs = n_jobs

    @property
    def estimators_(self):
        check_is_fitted(self, 'forest_')

        return self.forest_.estimators_

    @property
    def oob_predictions_(self):
        check_is_fitted(self, 'forest_')

        if not self.oob_mse:
            raise ValueError('OOB predictions were not calculated. '
                             'Set oob_mse=True during initialization.')

        return self.forest_.oob_predictions

    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape

        # check input arrays
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True,
                         dtype='float64')
        if y.dtype != np.float64:
            y = y.astype('float64')

        # check parameters
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto" :
                max_features = n_features
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features) + 1)
            elif self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
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

        if isinstance(self.min_samples_leaf, six.string_types):
            if self.min_samples_leaf == "auto":
                min_samples_leaf =  1
            else:
                raise ValueError("Unrecognized value for min_samples_leaf")
        elif isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
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

        self.forest_ = dimension_reduction_forest(
            X, y, num_trees=self.n_estimators,
            max_features=max_features,
            num_slices=self.n_slices,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            oob_error=self.oob_mse,
            n_jobs=self.n_jobs, seed=self.random_state)

        if self.oob_mse:
            self.oob_mse_ = mean_squared_error(y, self.oob_predicitons_)

        # save training data for kernel estimates
        if self.store_X_y:
            self.X_fit_ = X
            self.y_fit_ = y

        return self

    def _compute_kernel(self, X, Y=None):
        if not self.store_X_y:
            raise ValueError("Cannot compute the forest kernel. Set "
                             "`store_X_y=True` and re-fit the model.")

        X_leaves = self.apply(X)
        Y_leaves = Y if Y is None else self.apply(Y)
        return leaf_node_kernel(X_leaves, Y_leaves)

    def __call__(self, X, Y=None):
        check_is_fitted(self)

        X = np.atleast_2d(X)
        X = check_array(X, dtype='float64')
        if X.shape[1] != self.X_fit_.shape[1]:
            raise ValueError('Number of features of `X` = {}'
                             ', while number of features in the training data '
                             'was {}. These should be equal.'.format(
                                X.shape[1], self.X_fit_.shape[1]))

        if Y is not None:
            Y = check_array(Y, dtype='float64')
        else:
            Y = self.X_fit_

        return self._compute_kernel(X, Y)

    def predict(self, X, pred_type='average'):
        check_is_fitted(self)
        X = check_array(X, dtype='float64')

        if pred_type not in ['kernel', 'average']:
            raise ValueError("pred_type should be one of "
                             "{'kernel', 'average'}")

        if pred_type == 'average':
            return self.forest_.predict(X, self.n_jobs)

        K = self._compute_kernel(X, self.X_fit_)

        return np.dot(K, self.y_fit_) / np.sum(K, axis=1)

    def apply(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype='float64')

        return self.forest_.apply(X, self.n_jobs)

    def local_subspace_importances(self, X, n_jobs=1, n_directions=1):
        check_is_fitted(self)

        # must be a 2d array (n_samples, n_features)
        X = np.atleast_2d(X)
        X = check_array(X, dtype='float64')

        # extract kernel weights on in-sample data points
        weights = self(X, self.X_fit_)

        directions = Parallel(n_jobs=n_jobs)(
            delayed(local_direction)(
            X[i], self.X_fit_, weights[i], n_directions) for
            i in range(X.shape[0]))

        return np.squeeze(np.asarray(directions))
