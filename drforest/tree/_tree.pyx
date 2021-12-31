# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cpython.ref cimport Py_INCREF, PyTypeObject, PyObject
from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject *subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)


NODE_DTYPE = np.dtype({
    'names': ['node_id', 'feature', 'depth', 'left_child', 'right_child',
              'threshold', 'impurity', 'value', 'num_node_samples',
              'weighted_n_node_samples'],
    'formats': [np.uintp, np.intp, np.uintp, np.intp, np.intp,
                np.float64, np.float64, np.float64, np.uintp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).node_id,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).depth,
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).value,
        <Py_ssize_t> &(<Node*> NULL).num_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples,
    ]
})


cdef class DimensionReductionTree:
    """Array-based representation of a binary dimension reduction tree."""
    def __init__(self, *args, **kwargs):
        raise TypeError("Cannot create instance from Python")

    @staticmethod
    cdef DimensionReductionTree init(shared_ptr[Tree] tree_ptr):
        obj = <DimensionReductionTree>DimensionReductionTree.__new__(
            DimensionReductionTree)
        obj.tree = tree_ptr
        return obj

    property directions:
        def __get__(self):
            return self._get_directions_ndarray()

    property node_id:
        def __get__(self):
            return self._get_node_ndarray()['node_id']

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature']

    property node_depth:
        def __get__(self):
            return self._get_node_ndarray()['depth']

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child']

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child']

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold']

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity']

    property value:
        def __get__(self):
            return self._get_node_ndarray()['value']

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['num_node_samples']

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples']

    property node_count:
        def __get__(self):
            return deref(self.tree).num_nodes()

    def apply(self, np.ndarray[np.double_t, ndim=2] X):
        cdef mat X_mat = to_arma_mat(X)
        cdef uvec indices

        indices = deref(self.tree).apply(X_mat)

        return to_1d_uint_ndarray(indices)

    def predict(self, np.ndarray[np.double_t, ndim=2] X):
        cdef mat X_mat = to_arma_mat(X)
        cdef vec preds

        preds = deref(self.tree).predict(X_mat)

        return to_1d_ndarray(preds)

    def decision_path(self, np.ndarray[np.double_t, ndim=2] X):
        cdef mat X_mat = to_arma_mat(X)
        cdef umat out

        out = deref(self.tree).decision_path(X_mat)

        return to_uint_ndarray(out)

    def generate_oob_indices(self):
        cdef uvec out

        out = deref(self.tree).generate_oob_indices(True)

        return to_1d_uint_ndarray(out)

    def estimate_sufficient_dimensions(self,
                                       np.ndarray[np.double_t, ndim=2] X,
                                       int algorithm=0):
        cdef mat X_mat = to_arma_mat(X)
        cdef mat out
        cdef DimensionReductionAlgorithm dr_algorithm

        if algorithm == 0:
            dr_algorithm = DimensionReductionAlgorithm.SIR
        else:
            dr_algorithm = DimensionReductionAlgorithm.SAVE

        out = deref(self.tree).estimate_sufficient_dimensions(X_mat,
                                                              dr_algorithm)

        return to_ndarray(out)

    cdef np.ndarray _get_node_ndarray(self):
        cdef Node* nodes_ptr = deref(self.tree).get_nodes_ptr()

        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> deref(self.tree).num_nodes()
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject*> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> nodes_ptr,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self

        return arr

    cdef np.ndarray _get_directions_ndarray(self):
        cdef double* array_ptr = deref(self.tree).get_directions_ptr()

        # Armadillo stores matrices in column-major order, but numpy thinks
        # data is stored in row-major order. The result is that the matrix
        # is transposed on the numpy side (n_directions, n_features)
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> deref(self.tree).get_directions().n_cols
        shape[1] = <np.npy_intp> deref(self.tree).get_directions().n_rows

        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, array_ptr)
        Py_INCREF(self)
        arr.base = <PyObject*> self

        return arr

    def get_oob_indices(self):
        cdef uvec oob_indices = deref(self.tree).generate_oob_indices(True)

        return to_1d_uint_ndarray(oob_indices)


def dimension_reduction_tree(np.ndarray[np.double_t, ndim=2] X,
                             np.ndarray[np.double_t, ndim=1] y,
                             np.ndarray[np.double_t, ndim=1] sample_weight,
                             np.ndarray[np.int_t, ndim=1] numeric_features,
                             np.ndarray[np.int_t, ndim=1] categorical_features,
                             int num_slices=10,
                             int max_features=-1,
                             int max_depth=10,
                             int min_samples_leaf=2,
                             bool use_original_features=False,
                             bool presorted=False,
                             int seed=42):
    cdef mat X_mat = to_arma_mat(X)
    cdef vec y_vec = to_arma_vec(y)
    cdef vec sample_weight_vec = to_arma_vec(sample_weight)
    cdef uvec num_features = to_arma_uvec(numeric_features)
    cdef uvec cat_features = to_arma_uvec(categorical_features)
    cdef shared_ptr[Tree] tree

    tree = build_dimension_reduction_tree(X_mat, y_vec, sample_weight_vec,
                                          num_features, cat_features,
                                          max_features,
                                          num_slices, max_depth,
                                          min_samples_leaf,
                                          use_original_features,
                                          presorted,
                                          seed)

    return DimensionReductionTree.init(tree)
