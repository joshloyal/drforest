import numbers

import numpy as np

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._forest import dimension_reduction_forest
from .feature_importance import permutation_importance

__all__ = ['DimensionReductionForestRegressor', 'error_curve']


def error_curve(forest, X, y):
    n_estimators = forest.n_estimators

    y_pred = np.zeros(X.shape[0])
    err = np.zeros(n_estimators)
    for i in range(n_estimators):
        tree_y_pred = forest.estimators_[i].predict(X)
        y_pred += (tree_y_pred - y_pred) / (i + 1)
        err[i] = mean_squared_error(y, y_pred)

    return err


def leaf_node_kernel(X_leaves, Y_leaves=None):
    """Random forest kernel function."""
    return 1 - pairwise_distances(X_leaves, Y=Y_leaves, metric='hamming')


def local_direction(x0, X_train, weights, n_directions=1):
    """Calculate the local principal direction at x0."""
    # filter for data points with non-zero weights
    nonzero = weights != 0
    X_nonzero = X_train[nonzero] - x0

    # calculate the local PCA (bandwidth) matrix
    w = weights[nonzero].reshape(-1, 1)
    X_nonzero -= (w * X_nonzero).sum(axis=0) / np.sum(w)
    M = np.dot(X_nonzero.T, X_nonzero * w) / np.sum(w)

    eigval, eigvec = np.linalg.eigh(M)

    return eigvec[:, :n_directions]


class DimensionReductionForestRegressor(BaseEstimator, RegressorMixin):
    """Dimension Reduction Forest Regressor.

    A dimension reduction forest (DRF) is a random forest [1] composed of
    dimension reduction trees that use sufficient dimension reduction (SDR)
    techniques to approximate a locally adaptive kernel. Furthermore, DRFs
    leverage this adaptivity to estimate a local variable importance measure
    known as the local principal direction (LPD).

    Dimension reduction trees use a combinatoin of Sliced Inverse Regression
    (SIR) [2] and Sliced Average Variance Estimation (SAVE) [3] to estimate a
    linear combination splitting rule. The splitting criterion is
    mean squared error (variance) at each node.

    Parameters
    ----------
    n_estimators : int, optional (default=500)
        The number of trees in the forest.

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

    min_samples_leaf : int, float, optional (default=3)
        The minimum number of samples required to be a leaf node. A split
        point at any depth will only be considered if it leaves at least
        ``min_samples_leaf`` training samples in each leaf and right branches.

    n_slices : int, optional (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    oob_mse : bool, optional (default=True)
        Whether to use out-of-bag samples to estimate the MSE of unseen data.

    store_X_y : bool, optional(default=True)
        Whether to store the training data X, y. This is required for
        calculating the random forest kernel.

    random_state : int, optional (default=42)
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the randomness inside the tree building process.

    n_jobs: int, optional(default=1)
        The number of jobs to run in parallel. ``fit``, ``predict`, and
        ``apply`` are all parallelized over the trees.

    Attributes
    ----------
    forest_ : DimensionReductionForest instance
        The underlying DimensionReductionForest object. This is a wrapper
        to the underlying c++ class and is not intended for general usage.

    estimators_ : list of DimensionReductionTree
        The collection of fitted dimension reduction trees. Please refer to
        ``drforest.tree._tree.DimensionReductionTree``.

    feature_importances_ : ndarray of shape (n_features,)
        The permutation-based feature importance based on out-of-bag
        predictions.

    oob_mse_ : float
        Mean squared error of the training dataset obtained using an
        out-of-bag estimate. The attribute exists only when `oob_mse` is
        True.

    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimates on the training set.
        This attribute is  only computable when ``oob_score` is True.

    X_fit_ : ndarray of shape (n_samples, n_features)
        The full training features. This attribute exists only when
        ``store_X_y`` is True.

    y_fit_ : ndarray of shape (n_features,)
        The full training response. This attribute exists only when
        ``store_X_y`` is True.

    Examples
    --------

    >>> from drforest.datasets import make_cubic
    >>> from drforest.ensemble import DimensionReductionForestRegressor
    >>> X, y = make_cubic(n_features=2, random_state=123)
    >>> drf = DimensionReductionForestRegressor(n_estimators=10)
    >>> drf.fit(X, y)
    >>> print(drf.predict([[-1, 1, 0]]))
    array([0.15467771])
    >>> print(drf.local_subspace_importance([[-1, 1, 0]]))
    array([-0.70722344, -0.70577232, -0.04147822])

    References
    ----------

    [1] Breiman (2001)
        "Random Forests", Machine Learning, 45(1), 5-32.

    [2] Li, K C. (1991)
        "Sliced Inverse Regression for Dimension Reduction (with discussion)",
        Journal of the American Statistical Association, 86, 316-342.

    [3] Shao, Y, Cook, RD and Weisberg, S (2007).
        "Marginal Tests with Sliced Average Variance Estimation",
        Biometrika, 94, 285-296.
    """
    def __init__(self,
                 n_estimators=500,
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=3,
                 n_slices=10,
                 oob_mse=True,
                 store_X_y=True,
                 random_state=42,
                 n_jobs=1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_slices = n_slices
        self.oob_mse = oob_mse
        self.random_state = random_state
        self.store_X_y = store_X_y
        self.n_jobs = n_jobs

    @property
    def estimators_(self):
        """The collection of fitted dimension reduction trees.

        Returns
        -------
        estimators_ : list of length n_estimators
            The collection of fitted dimension reduction trees.
        """
        check_is_fitted(self, 'forest_')

        return self.forest_.estimators_

    @property
    def oob_prediction_(self):
        """Predictions computed with out-of-bag estimates on the training set.

        Returns
        -------
        oob_predictions_ : ndarray of shape (n_samples,)
           Prediction computed with out-of-bag estimate on the training set.
        """
        check_is_fitted(self, 'forest_')

        if not self.oob_mse:
            raise ValueError('OOB predictions were not calculated. '
                             'Set oob_mse=True during initialization.')

        return self.forest_.oob_predictions

    @property
    def feature_importances_(self):
        """Out-of-bag permutation-based feature importances.

        The higher, the more important the feature. The importances are
        normalized to sum to one.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1.
        """
        check_is_fitted(self, 'forest_')

        # FIXME: currently does not support multiprocessing, since objects
        # are not serializable.
        all_importances = permutation_importance(
            self, self.X_fit_, self.y_fit_, random_state=self.random_state)

        return all_importances / np.sum(all_importances)

    def fit(self, X, y, sample_weight=None):
        """Build a dimension reduction forest from the training set (X, y)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape

        # check input arrays
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True,
                         dtype='float64')
        if y.dtype != np.float64:
            y = y.astype('float64')

        # check parameters
        max_depth = -1 if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, str):
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

        if isinstance(self.min_samples_leaf, str):
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
            self.oob_mse_ = mean_squared_error(y, self.oob_prediction_)

        # save training data for kernel estimates
        if self.store_X_y:
            self.X_fit_ = X
            self.y_fit_ = y

        return self

    def __call__(self, X, Y=None):
        """Compute the induced random forest kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples_a, n_features)
            A feature array.
        Y : ndarray of shape (n_samples_b, n_features)
            A second feature array.

        Returns
        -------
        K : ndarray of shape (n_samples_a, n_samples_a) or \
                (n_samples_a, n_samples_b)
            The random forest kernel matrix K, such that K_{i, j} is the
            kernel value between the ith and jth vectors of the given matrix X, if
            Y is None. If Y is not None, then K_{i, j} is the distance between the
            ith vector from X and the jth vector from Y.
        """
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
            if not self.store_X_y:
                raise ValueError("Cannot compute the forest kernel. Set "
                                 "`store_X_y=True` and re-fit the model.")
            Y = self.X_fit_

        X_leaves = self.apply(X)
        Y_leaves = Y if Y is None else self.apply(Y)
        return leaf_node_kernel(X_leaves, Y_leaves)

    def predict(self, X, pred_type='average'):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest or as a
        Nadarya-Watson kernel estimate using the induced random forest kernel.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        pred_type : string in {'average', 'kernel'}
            The type of predictions. The averge over trees in the forest when
            pred_type is 'average', and a kernel estimate when pred_type is
            'kernel'.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
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
        """Apply trees in the forest to X, return leaf indices.

        Paramters
        ---------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest, return
            the index of the leaf x ends up in.
        """
        check_is_fitted(self)
        X = check_array(X, dtype='float64')

        return self.forest_.apply(X, self.n_jobs)

    def local_principal_direction(self, X, n_jobs=1):
        """Calculate the local principal direction at each point in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        n_jobs: int, optional(default=1)
            The number of jobs to run in parallel. The calculation is
            parallelized over the samples.

        Returns
        -------
        local_directions : ndarray of shape (n_samples, n_features)
            The local principal direction (LPD) at each point x in X.
            An LPD is interpreted as the one-dimensional subpsace that most
            influences the regression function at x.
        """
        check_is_fitted(self)

        # must be a 2d array (n_samples, n_features)
        X = np.atleast_2d(X)
        X = check_array(X, dtype='float64')

        # extract kernel weights on in-sample data points
        weights = self(X, self.X_fit_)

        directions = Parallel(n_jobs=n_jobs)(
            delayed(local_direction)(
            X[i], self.X_fit_, weights[i]) for
            i in range(X.shape[0]))

        return np.squeeze(np.asarray(directions))
