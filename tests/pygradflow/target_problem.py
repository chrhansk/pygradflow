import numpy as np
import scipy as sp

from pygradflow.problem import Problem

x0 = np.array([-1.0, 1.0])
x1 = np.array([1.0, -1.0])

optima = [x0, x1]


class TargetProblem(Problem):
    def __init__(self):
        var_lb = np.array([-np.inf, -np.inf])
        var_ub = np.array([np.inf, np.inf])
        super().__init__(var_lb, var_ub, num_cons=0)

    def obj(self, x):
        return np.dot(x - x0, x - x0) * np.dot(x - x1, x - x1)

    def obj_grad(self, x):
        return 2 * (x - x0) * np.dot(x - x1, x - x1) + 2 * (x - x1) * np.dot(
            x - x0, x - x0
        )

    def lag_hess(self, x, y):
        mat = (2 * (np.dot(x - x1, x - x1) + np.dot(x - x0, x - x0))) * np.eye(2)
        mat += 4 * np.outer(x - x0, x - x1)
        mat += 4 * np.outer(x - x1, x - x0)

        return sp.sparse.csc_matrix(mat)
