# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np


cdef mat to_arma_mat(np.ndarray[np.double_t, ndim=2] np_array):
    """Converts a numpy ndarray to an arma::mat. A copy of the array  is made
    if it either does not own its data or that data is stored in
    c-ordered format. Note that c-ordered is the default format
    for numpy arrays.
    """
    if np_array.flags.c_contiguous or not np_array.flags.owndata:
        np_array = np_array.copy(order='F')

    return mat(<double*>np_array.data,
               np_array.shape[0], np_array.shape[1], False, True)


cdef np.ndarray[np.double_t, ndim=2] to_ndarray(const mat& arma_mat):
    """Converts an arma::mat to a numpy ndarray. Currently, a
    copy is always made.
    """
    cdef int i = 0
    cdef n_rows = arma_mat.n_rows
    cdef n_cols = arma_mat.n_cols
    cdef const double* mat_ptr
    cdef double* np_ptr
    cdef np.ndarray[np.double_t, ndim=2] np_array

    # allocate memory for the new np.array
    np_array = np.empty((n_rows, n_cols),
                        dtype=np.double,
                        order="F")

    # copy data from the arma::mat to the numpy array
    mat_ptr = arma_mat.memptr()
    np_ptr = <double*> np_array.data
    for i in range(n_rows * n_cols):
        np_ptr[i] = mat_ptr[i]

    return np_array


cdef np.ndarray[np.int_t, ndim=2] to_uint_ndarray(const umat& arma_mat):
    """Converts an arma::mat to a numpy ndarray. Currently, a
    copy is always made.
    """
    cdef int i = 0
    cdef n_rows = arma_mat.n_rows
    cdef n_cols = arma_mat.n_cols
    cdef const uword* mat_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=2] np_array

    # allocate memory for the new np.array
    np_array = np.empty((n_rows, n_cols),
                        dtype=np.int,
                        order="F")

    # copy data from the arma::mat to the numpy array
    mat_ptr = arma_mat.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_rows * n_cols):
        np_ptr[i] = mat_ptr[i]

    return np_array

cdef vec to_arma_vec(np.ndarray[np.double_t, ndim=1] np_array):
    """Converts a 1d numpy array to an arma::vec. Data is copied if
    the array does not own its data.
    """
    if not np_array.flags.owndata:
        np_array = np_array.copy(order='F')

    return vec(<double*> np_array.data, np_array.shape[0], False, True)


cdef np.ndarray[np.double_t, ndim=1] to_1d_ndarray(const vec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const double* vec_ptr
    cdef double* np_ptr
    cdef np.ndarray[np.double_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=np.double)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <double*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <double>vec_ptr[i]

    return np_array


cdef np.ndarray[np.int_t, ndim=1] to_1d_int_ndarray(const ivec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const sword* vec_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=np.int)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <np.int_t>vec_ptr[i]

    return np_array


cdef np.ndarray[np.int_t, ndim=1] to_1d_uint_ndarray(const uvec& arma_vec):
    """Converts an arma::vec to a 1d numpy array. Currently, a copy is
    always made.
    """
    cdef int i = 0
    cdef n_elem = arma_vec.n_elem
    cdef const uword* vec_ptr
    cdef np.int_t* np_ptr
    cdef np.ndarray[np.int_t, ndim=1] np_array

    # allocate memory for the new np.array
    np_array = np.empty(n_elem, dtype=np.int)

    # copy data from the arma::vec to the numpy array
    vec_ptr = arma_vec.memptr()
    np_ptr = <np.int_t*> np_array.data
    for i in range(n_elem):
        np_ptr[i] = <np.int_t>vec_ptr[i]

    return np_array
