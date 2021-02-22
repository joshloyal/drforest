# cython: language_level=3

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from drforest.armadillo cimport mat, umat, vec
from drforest.armadillo cimport to_arma_mat, to_uint_ndarray, to_arma_vec
from drforest.armadillo cimport to_1d_ndarray, to_ndarray

from drforest.tree._tree cimport Tree, DimensionReductionTree

cdef extern from "drforest.h" namespace "drforest" nogil:
    cdef cppclass RandomForest:
        vector[shared_ptr[Tree]]& get_trees()

        shared_ptr[Tree] get_tree(size_t i)

        vec predict(mat &X, int num_threads)

        umat apply(mat &X, int num_threads)

        vec get_oob_predictions()

        double get_oob_error()

        mat estimate_sufficient_dimensions(mat &X, int num_threads)

    shared_ptr[RandomForest] train_random_forest(mat& X,
                                                 vec& y,
                                                 size_t num_trees,
                                                 int max_features,
                                                 size_t num_slices,
                                                 int max_depth,
                                                 size_t min_samples_leaf,
                                                 bool oob_error,
                                                 int num_threads,
                                                 int seed)

    shared_ptr[RandomForest] train_permuted_random_forest(
        mat& X,
        vec& y,
        unsigned int feature_id,
        size_t num_trees,
        int max_features,
        size_t num_slices,
        int max_depth,
        size_t min_samples_leaf,
        bool oob_error,
        int num_threads,
        int seed)
