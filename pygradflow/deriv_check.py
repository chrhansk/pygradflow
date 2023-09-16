from typing import Callable, Union

import numpy as np
import scipy as sp

from pygradflow.params import Params


def deriv_check(
    f: Callable,
    xval: np.ndarray,
    dval: Union[np.ndarray, sp.sparse.spmatrix],
    params: Params,
) -> None:
    (n,) = xval.shape

    fval = f(xval)

    fval = np.atleast_1d(fval)

    eps = params.deriv_pert

    dsparse = False

    if sp.sparse.issparse(dval):
        dval = dval.tocsc()
        dsparse = True
    else:
        dval = np.atleast_2d(dval)

    xtest = np.copy(xval)

    for i in range(n):
        xtest[i] += eps

        testval = f(xtest)

        apx_dval = (testval - fval) / eps

        darray = dval[:, i]

        if dsparse:
            darray = darray.toarray()

        assert np.allclose(darray, apx_dval[:, None], atol=params.deriv_tol)

        xtest[i] -= eps
