import numpy as np
import scipy.sparse

from pygradflow.problem import Problem


class Tame(Problem):
    def __init__(self):
        lb = np.array([-np.inf, -np.inf])
        ub = np.array([np.inf, np.inf])
        super().__init__(lb, ub, num_cons=1)

    def obj(self, z):
        (x, y) = z

        return (x - y) ** 2

    def obj_grad(self, z):
        (x, y) = z
        return np.array([2 * (x - y), -2 * (x - y)])

    def cons(self, z):
        (x, y) = z
        return np.array([x + y - 1])

    def cons_jac(self, z):
        return scipy.sparse.coo_matrix(np.array([[1, 1]]))

    def lag_hess(self, z, lag):
        obj_hess = np.array([[2.0, -2.0], [-2.0, 2.0]])

        h1 = np.array([[0, 0], [0, 0]])

        return scipy.sparse.coo_matrix(obj_hess + lag[0] * h1)
