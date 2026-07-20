import numpy as np
import scipy as sp

from pygradflow.problem import Problem


class LinearProgram(Problem):
    """
    Linear program: minimize c^T x subject to Ax <= b
    """

    def __init__(self, A, b, c):
        (num_cons, num_vars) = A.shape

        var_lb = np.full(shape=(num_vars,), fill_value=-np.inf)
        var_ub = np.full(shape=(num_vars,), fill_value=np.inf)

        cons_lb = np.full(shape=(num_cons,), fill_value=-np.inf)
        cons_ub = b

        self.c = c
        self.b = b
        self.A = sp.sparse.coo_matrix(A)

        super().__init__(var_lb, var_ub, cons_lb=cons_lb, cons_ub=cons_ub)

    def obj(self, x):
        return np.dot(self.c, x)

    def obj_grad(self, x):
        return self.c

    def cons(self, x):
        return self.A.dot(x)

    def cons_jac(self, x):
        return self.A

    def lag_hess(self, x, y):
        return sp.sparse.coo_matrix(
            np.zeros(
                (
                    self.num_vars,
                    self.num_vars,
                ),
                dtype=float,
            )
        )
