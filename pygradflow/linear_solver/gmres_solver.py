import numpy as np
import scipy as sp

from .linear_solver import LinearSolver, LinearSolverError


class GMRESSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix, symmetric=False) -> None:
        super().__init__(mat, symmetric=symmetric)
        self.mat = mat

    def solve(self, rhs, trans=False, initial_sol=None):
        mat = self.mat.T if trans else self.mat

        if initial_sol is not None:
            initial_sol = initial_sol()

        (n, _) = mat.shape

        atol = 1e-8

        # Workaround for scipy bug
        if initial_sol is not None:
            res = rhs - mat @ initial_sol
            if np.linalg.norm(res, ord=np.inf) < atol:
                return initial_sol

        result = sp.sparse.linalg.gmres(mat, rhs, maxiter=n, x0=initial_sol, atol=atol)

        (sol, info) = result

        if info != 0:
            raise LinearSolverError("GMRES failed with error code {}".format(info))

        return sol
