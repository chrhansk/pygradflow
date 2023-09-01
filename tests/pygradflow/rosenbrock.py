import numpy as np
import scipy

from pygradflow.problem import Problem


class Rosenbrock(Problem):
    def __init__(self):
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([np.inf, np.inf])
        super().__init__(lb, ub)
        self.a = 1.0
        self.b = 100.0

    def obj(self, v):
        [x, y] = v
        a = self.a
        b = self.b

        return (a - x) ** 2 + b * (y - x**2) ** 2

    def obj_grad(self, v):
        [x, y] = v
        a = self.a
        b = self.b
        return np.array(
            [4 * (x**2 - y) * b * x - 2 * a + 2 * x, -2 * (x**2 - y) * b]
        )

    def cons(self, v):
        return np.array([])

    def cons_jac(self, v):
        return scipy.sparse.coo_matrix(np.zeros((0, 2)))

    def lag_hess(self, v, _):
        [x, y] = v

        a = self.a
        b = self.b

        h = np.array(
            [
                [8 * b * x**2 + 4 * (x**2 - y) * b + 2, -4 * b * x],
                [-4 * b * x, 2 * b],
            ]
        )

        return scipy.sparse.coo_matrix(h)
