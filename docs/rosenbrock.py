import numpy as np
import scipy as sp

from pygradflow.problem import Problem


class Rosenbrock(Problem):
    def __init__(self):
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([np.inf, np.inf])
        super().__init__(lb, ub)
        self.a = 1.0
        self.b = 100.0

    def obj(self, x):
        [x0, x1] = x
        a = self.a
        b = self.b

        return (a - x0) ** 2 + b * (x1 - x0**2) ** 2

    def obj_grad(self, x):
        [x0, x1] = x
        a = self.a
        b = self.b
        return np.array(
            [4 * (x0**2 - x1) * b * x0 - 2 * a + 2 * x0, -2 * (x0**2 - x1) * b]
        )

    def cons(self, x):
        return np.array([])

    def cons_jac(self, x):
        return sp.sparse.coo_matrix(np.zeros((0, 2)))

    def lag_hess(self, x, _):
        [x0, x1] = x

        b = self.b

        h = np.array(
            [
                [8 * b * x0**2 + 4 * (x0**2 - x1) * b + 2, -4 * b * x0],
                [-4 * b * x0, 2 * b],
            ]
        )

        return sp.sparse.coo_matrix(h)


import logging

import numpy as np

from pygradflow.solver import Solver

logging.basicConfig(level=logging.INFO)

rosenbrock = Rosenbrock()
solver = Solver(problem=rosenbrock)

solution = solver.solve()

print(solution.x)
print(solution.status)
