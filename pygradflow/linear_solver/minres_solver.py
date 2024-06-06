import scipy as sp

from .linear_solver import LinearSolver, LinearSolverError


class MINRESSolver(LinearSolver):
    def __init__(self, mat, symmetric=False):
        super().__init__(mat, symmetric=symmetric)
        assert symmetric, "MINRES requires a symmetric matrix"
        self.mat = mat

    def solve(self, rhs, trans=False, initial_sol=None):
        # matrix should be symmetric anyways
        if initial_sol is not None:
            initial_sol = initial_sol()

        result = sp.sparse.linalg.minres(self.mat, rhs, x0=initial_sol)

        (sol, info) = result

        if info != 0:
            raise LinearSolverError("MINRES failed with error code {}".format(info))

        return sol
