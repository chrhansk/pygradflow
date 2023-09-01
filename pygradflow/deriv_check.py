import numpy as np
import scipy


def deriv_check(f, xval, dval, params):
    (n,) = xval.shape

    fval = f(xval)

    fval = np.atleast_1d(fval)

    eps = params.deriv_pert

    dsparse = False

    if scipy.sparse.issparse(dval):
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
