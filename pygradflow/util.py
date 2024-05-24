import numpy as np
import scipy as sp
from numpy import ndarray


def sparse_zero(shape, format=None):
    zero_mat = sp.sparse.coo_matrix(([], ([], [])), shape)

    if format not in [None, "coo"]:
        return zero_mat.asformat(format)

    return zero_mat


def norm_sq(x: ndarray) -> float:
    return np.dot(x, x)


def norm_mult(*args) -> float:
    value = 0.0
    for arg in args:
        value += norm_sq(arg)

    return np.sqrt(value)


def keep_rows(mat, row_filter):
    (m, _) = mat.shape

    assert row_filter.shape == (m,)
    assert row_filter.dtype == bool

    if row_filter.all():
        return mat

    mat = mat.tocoo()

    rows = mat.row
    cols = mat.col
    data = mat.data

    nnz = mat.nnz
    indices = np.arange(nnz)

    index_filter = row_filter[rows[indices]]

    alive_rows = rows[index_filter]
    alive_cols = cols[index_filter]
    alive_data = data[index_filter]

    cleared_mat = sp.sparse.coo_matrix(
        (alive_data, (alive_rows, alive_cols)), shape=mat.shape
    )

    return cleared_mat
