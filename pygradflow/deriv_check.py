from typing import Callable, Union

import numpy as np
import scipy as sp

from pygradflow.params import Params


class DerivError(ValueError):
    def __init__(self, expected_value, actual_value, col_index, atol) -> None:
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.atol = atol

        self.invalid_deriv = np.isclose(self.expected_value,
                                        self.actual_value,
                                        atol=self.atol)

        self.invalid_deriv = np.logical_not(self.invalid_deriv)
        self.invalid_indices = np.where(self.invalid_deriv)[0]

        self.deriv_diffs = np.abs(self.expected_value - self.actual_value)
        self.max_deriv_diff = self.deriv_diffs.max()

        (num_rows, _) = self.deriv_diffs.shape

        invalid_indices = np.zeros((num_rows, 2), dtype=int)
        invalid_indices[:, 0] = self.invalid_indices
        invalid_indices[:, 1] = col_index

        self.invalid_indices = invalid_indices
        self.col_index = col_index

    def __str__(self):
        num_invalid_indices = self.invalid_indices.size

        message = (f"Expected derivative: {self.expected_value} "
                   f"and actual (findiff) derivative: {self.actual_value} "
                   f"differ at the {num_invalid_indices} "
                   f"indices: {self.invalid_indices} "
                   f"(max diff: {self.max_deriv_diff}, tolerance: {self.atol})")

        return message


def deriv_check(
    f: Callable,
    xval: np.ndarray,
    dval: Union[np.ndarray, sp.sparse.spmatrix],
    params: Params,
) -> None:
    (n,) = xval.shape

    fval = f(xval)
    fval = np.atleast_1d(fval)

    (m,) = fval.shape

    sparse_dval = sp.sparse.issparse(dval)

    eps = params.deriv_pert

    dsparse = False

    if sparse_dval:
        dval = dval.tocsc()
        dsparse = True
    else:
        dval = np.atleast_2d(dval)

    assert dval.shape == (m, n)

    xtest = np.copy(xval)

    for i in range(n):
        xtest[i] += eps

        testval = f(xtest)

        apx_dval = (testval - fval) / eps

        if apx_dval.ndim == 1:
            apx_dval = apx_dval[:, np.newaxis]

        darray = dval[:, i]

        if dsparse:
            darray = darray.toarray()

        darray = np.atleast_2d(darray)

        assert darray.shape == apx_dval.shape

        if not np.allclose(darray, apx_dval, atol=params.deriv_tol):
            raise DerivError(darray, apx_dval, i, params.deriv_tol)

        xtest[i] -= eps
