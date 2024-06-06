import sksparse.cholmod

from .linear_solver import LinearSolver, LinearSolverError


class CholeskySolver(LinearSolver):
    def __init__(self, mat, symmetric=False):
        super().__init__(mat, symmetric=symmetric)

        try:
            self.factor = sksparse.cholmod.cholesky(mat)
            self.factor.L()
        except sksparse.cholmod.CholmodNotPositiveDefiniteError as e:
            raise LinearSolverError() from e

    def solve(self, rhs, trans=False, initial_sol=None):
        assert not trans

        return self.factor.solve_A(rhs)

    def num_neg_eigvals(self):
        return 0
