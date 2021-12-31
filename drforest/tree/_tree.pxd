# cython: language_level=3

from libcpp cimport bool
from libc.stdint cimport uint64_t, int64_t
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from drforest.armadillo cimport mat, umat, vec, ivec, uvec
from drforest.armadillo cimport to_arma_mat, to_arma_vec, to_arma_uvec
from drforest.armadillo cimport to_ndarray, to_uint_ndarray, to_1d_ndarray
from drforest.armadillo cimport to_1d_int_ndarray, to_1d_uint_ndarray


cdef extern from "drforest.h" namespace "drforest" nogil:
    cdef enum DimensionReductionAlgorithm:
        SIR
        SAVE

    cdef struct Node:
        uint64_t node_id                # id of the node in the tree
        int64_t feature                 # id of direction or feature
        uint64_t depth                  # depth of node in the tree
        int64_t left_child              # id of the left child of the node
        int64_t right_child             # id of the right child of the node
        double threshold                # Threshold value at the node
        double impurity                 # Impurity of the node (i.e. the variance)
        double value                    # predicted value at the node (mean)
        uint64_t num_node_samples       # Number of samples at the node
        double weighted_n_node_samples  # Weighted number of samples at the node

    cdef cppclass Tree:
        Tree(size_t num_features)

        Node* get_nodes_ptr()
        double* get_directions_ptr()

        size_t num_nodes()
        const mat& get_directions()

        uvec apply(const mat& X)
        vec predict(const mat& X)
        umat decision_path(const mat& X)
        mat estimate_sufficient_dimensions(const mat& X,
                                           DimensionReductionAlgorithm dr_algo)
        uvec generate_oob_indices(bool return_unsorted_indices)


    # Builds a dimension dimension tree with data X, y, wample_weight
    # and with the various hyperparameters set
    shared_ptr[Tree] build_dimension_reduction_tree(mat& X,
                                                    vec& y,
                                                    vec& sample_weight,
                                                    uvec& numeric_features,
                                                    uvec& categorical_features,
                                                    int max_features,
                                                    int num_slices,
                                                    int max_depth,
                                                    size_t min_samples_leaf,
                                                    bool use_original_features,
                                                    bool presorted,
                                                    int seed)

cdef class DimensionReductionTree:
    cdef shared_ptr[Tree] tree

    @staticmethod
    cdef DimensionReductionTree init(shared_ptr[Tree] tree_ptr)

    cdef np.ndarray _get_node_ndarray(self)

    cdef np.ndarray _get_directions_ndarray(self)
