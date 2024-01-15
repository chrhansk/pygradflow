import numpy as np
import scipy.sparse

from pygradflow.problem import Problem


class HS71(Problem):
    """
    HS71 test problem from Hock-Schittkowski test suite.
    Contains an additional slack variable to make the nonlinear
    constraints equations.
    """

    def __init__(self):
        lb = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        ub = np.array([5.0, 5.0, 5.0, 5.0, np.inf])
        super().__init__(lb, ub, num_cons=2)

    def obj(self, x):
        xx = x[:-1]
        return xx[0] * xx[3] * (xx[0] + xx[1] + xx[2]) + xx[2]

    def obj_grad(self, x):
        xx = x[:-1]

        return np.array(
            [
                (xx[0] + xx[1] + xx[2]) * xx[3] + xx[0] * xx[3],
                xx[0] * xx[3],
                xx[0] * xx[3] + 1,
                (xx[0] + xx[1] + xx[2]) * xx[0],
                0,
            ]
        )

    def cons(self, x):
        xx = x[:-1]
        s = x[-1]
        return np.array([np.prod(xx) - s - 25.0, np.dot(xx, xx) - 40.0])

    def cons_jac(self, x):
        xx = x[:-1]

        jac = np.array(
            [
                [
                    xx[1] * xx[2] * xx[3],
                    xx[0] * xx[2] * xx[3],
                    xx[0] * xx[1] * xx[3],
                    xx[0] * xx[1] * xx[2],
                    -1,
                ],
                [2 * xx[0], 2 * xx[1], 2 * xx[2], 2 * xx[3], 0],
            ]
        )

        return scipy.sparse.coo_matrix(jac)

    def lag_hess(self, x, lag):
        [l1, l2] = lag
        xx = x[:-1]

        obj_hess = np.array(
            [
                [2 * x[3], x[3], x[3], 2 * x[0] + x[1] + x[2], 0],
                [x[3], 0, 0, x[0], 0],
                [x[3], 0, 0, x[0], 0],
                [2 * x[0] + x[1] + x[2], x[0], x[0], 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        h1 = np.array(
            [
                [0, xx[2] * xx[3], xx[1] * xx[3], xx[1] * xx[2], 0],
                [xx[2] * xx[3], 0, xx[0] * xx[3], xx[0] * xx[2], 0],
                [xx[1] * xx[3], xx[0] * xx[3], 0, xx[0] * xx[1], 0],
                [xx[1] * xx[2], xx[0] * xx[2], xx[0] * xx[1], 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        h2 = np.zeros((x.size, x.size))

        h2[np.diag_indices(xx.size)] = 2.0

        h = obj_hess + l1 * h1 + l2 * h2

        return scipy.sparse.coo_matrix(h)
