# cython: language_level=3

from libcpp cimport bool

from drforest.armadillo cimport mat, vec, uvec
from drforest.armadillo cimport to_arma_mat, to_arma_vec
from drforest.armadillo cimport to_ndarray, to_1d_ndarray, to_1d_int_ndarray


cdef extern from "drforest.h" namespace "drforest" nogil:
    mat fit_sliced_inverse_regression(const mat& X,
                                      const vec& y,
                                      const int num_slices)

    mat fit_sliced_average_variance_estimation(const mat& X,
                                               const vec& y,
                                               const int num_slices)
