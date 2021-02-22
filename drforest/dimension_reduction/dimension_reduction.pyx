# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()


__all__ = ['sliced_inverse_regression', 'sliced_average_variance_estimation']


def sliced_inverse_regression(np.ndarray[np.double_t, ndim=2] X,
                              np.ndarray[np.double_t, ndim=1] y,
                              int n_slices=10):
    """Fits a Sliced Inverse Regression Model.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The training data.

    y : np.ndarray, shape (n_samples,)
        The target values.

    n_slices : int, optional (default=10)
        The number of slices used to calculate the inverse regression curve.
        Truncated to at most the number of unique values of `y`.

    Returns
    -------
    directions : np.ndarray, shape (n_directions, n_features)
        The directions in feature space representing the central subspace
        which is sufficient to describe the conditional distribution of
        y|X. The directions are sorted in decreasing order by their
        eigenvalues. Each row corresponds to direction.
    """
    cdef mat directions
    cdef mat X_mat = to_arma_mat(X)
    cdef vec y_vec = to_arma_vec(y)

    directions = fit_sliced_inverse_regression(X_mat, y_vec, n_slices)

    return to_ndarray(directions)


def sliced_average_variance_estimation(np.ndarray[np.double_t, ndim=2] X,
                                       np.ndarray[np.double_t, ndim=1] y,
                                       int n_slices=10):
    """Fits a Sliced Average Variance Estimation Model.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The training data.

    y : np.ndarray, shape (n_samples,)
        The target values.

    n_slices : int, optional (default=10)
        The number of slices used to calculate the inverse regression curve.
        Truncated to at most the number of unique values of `y`.

    Returns
    -------
    directions : np.ndarray, shape (n_directions, n_features)
        The directions in feature space representing the central subspace
        which is sufficient to describe the conditional distribution of
        y|X. The directions are sorted in decreasing order by their
        eigenvalues. Each row corresponds to direction.
    """
    cdef mat directions
    cdef mat X_mat = to_arma_mat(X)
    cdef vec y_vec = to_arma_vec(y)

    directions = fit_sliced_average_variance_estimation(X_mat, y_vec, n_slices)

    return to_ndarray(directions)
