# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np
np.import_array()


cdef class DimensionReductionForest:
    cdef shared_ptr[RandomForest] forest

    def __init__(self, *args, **kwargs):
        raise TypeError("Cannot create instance from Python")

    property estimators_:
        def __get__(self):
            return self._get_trees()

    property oob_predictions:
        def __get__(self):
            cdef vec oob_preds
            oob_preds = deref(self.forest).get_oob_predictions()

            return to_1d_ndarray(oob_preds)

    property oob_error:
        def __get__(self):
            return deref(self.forest).get_oob_error()

    @staticmethod
    cdef DimensionReductionForest init(shared_ptr[RandomForest] forest_ptr):
        obj = <DimensionReductionForest>DimensionReductionForest.__new__(
            DimensionReductionForest)
        obj.forest = forest_ptr
        return obj

    def predict(self, np.ndarray[np.double_t, ndim=2] X, int num_threads=1):
        cdef mat X_mat = to_arma_mat(X)
        cdef vec preds

        preds = deref(self.forest).predict(X_mat, num_threads)

        return to_1d_ndarray(preds)

    def apply(self, np.ndarray[np.double_t, ndim=2] X, int num_threads=1):
        cdef mat X_mat = to_arma_mat(X)
        cdef umat X_leaves

        X_leaves = deref(self.forest).apply(X_mat, num_threads)

        return to_uint_ndarray(X_leaves)

    def estimate_sufficient_dimensions(self,
                                       np.ndarray[np.double_t, ndim=2] X,
                                       int num_threads=1):
        cdef mat X_mat = to_arma_mat(X)
        cdef mat out

        out = deref(self.forest).estimate_sufficient_dimensions(X_mat,
                                                                num_threads)

        return to_ndarray(out)

    cdef list _get_trees(self):
        cdef size_t i = 0
        cdef vector[shared_ptr[Tree]] trees = deref(self.forest).get_trees()
        cdef size_t n_trees = trees.size()
        cdef list estimators = []

        for i in range(n_trees):
            estimators.append(DimensionReductionTree.init(trees.at(i)))

        return estimators


def dimension_reduction_forest(np.ndarray[np.double_t, ndim=2] X,
                               np.ndarray[np.double_t, ndim=1] y,
                               int num_trees=50,
                               int max_features=-1,
                               int num_slices=10,
                               int max_depth=-1,
                               int min_samples_leaf=2,
                               bool oob_error=False,
                               int n_jobs=-1,
                               int seed=42):
    cdef mat X_mat = to_arma_mat(X)
    cdef vec y_vec = to_arma_vec(y)
    cdef shared_ptr[RandomForest] forest

    forest = train_random_forest(X_mat, y_vec, num_trees, max_features,
                                 num_slices, max_depth, min_samples_leaf,
                                 oob_error, n_jobs, seed)

    return DimensionReductionForest.init(forest)


def permuted_dimension_reduction_forest(np.ndarray[np.double_t, ndim=2] X,
                                        np.ndarray[np.double_t, ndim=1] y,
                                        unsigned int feature_id,
                                        int num_trees=50,
                                        int max_features=-1,
                                        int num_slices=10,
                                        int max_depth=-1,
                                        int min_samples_leaf=2,
                                        bool oob_error=False,
                                        int n_jobs=-1,
                                        int seed=42):
    cdef mat X_mat = to_arma_mat(X)
    cdef vec y_vec = to_arma_vec(y)
    cdef shared_ptr[RandomForest] forest

    forest = train_permuted_random_forest(
        X_mat, y_vec, feature_id, num_trees, max_features,
        num_slices, max_depth, min_samples_leaf,
        oob_error, n_jobs, seed)

    return DimensionReductionForest.init(forest)
