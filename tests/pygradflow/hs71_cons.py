import numpy as np
import scipy.sparse

from pygradflow.problem import Problem


class HS71Constrained(Problem):
    """
    HS71 test problem from Hock-Schittkowski test suite.
    """

    def __init__(self):
        var_lb = np.array([1.0, 1.0, 1.0, 1.0])
        var_ub = np.array([5.0, 5.0, 5.0, 5.0])
        cons_lb = np.array([25.0, 40.0])
        cons_ub = np.array([np.inf, 40.0])
        super().__init__(var_lb, var_ub, cons_lb=cons_lb, cons_ub=cons_ub)

    def obj(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def obj_grad(self, x):
        return np.array(
            [
                (x[0] + x[1] + x[2]) * x[3] + x[0] * x[3],
                x[0] * x[3],
                x[0] * x[3] + 1,
                (x[0] + x[1] + x[2]) * x[0],
            ]
        )

    def cons(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def cons_jac(self, x):
        jac = np.array(
            [
                [
                    x[1] * x[2] * x[3],
                    x[0] * x[2] * x[3],
                    x[0] * x[1] * x[3],
                    x[0] * x[1] * x[2],
                ],
                [2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]],
            ]
        )

        return scipy.sparse.coo_matrix(jac)

    def lag_hess(self, x, lag):
        [l1, l2] = lag

        obj_hess = np.array(
            [
                [2 * x[3], x[3], x[3], 2 * x[0] + x[1] + x[2]],
                [x[3], 0, 0, x[0]],
                [x[3], 0, 0, x[0]],
                [2 * x[0] + x[1] + x[2], x[0], x[0], 0],
            ]
        )

        h1 = np.array(
            [
                [0, x[2] * x[3], x[1] * x[3], x[1] * x[2]],
                [x[2] * x[3], 0, x[0] * x[3], x[0] * x[2]],
                [x[1] * x[3], x[0] * x[3], 0, x[0] * x[1]],
                [x[1] * x[2], x[0] * x[2], x[0] * x[1], 0],
            ]
        )

        h2 = np.zeros((x.size, x.size))

        h2[np.diag_indices(x.size)] = 2.0

        h = obj_hess + l1 * h1 + l2 * h2

        return scipy.sparse.coo_matrix(h)
