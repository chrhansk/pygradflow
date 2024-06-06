import scipy as sp

from pygradflow.log import logger

from .linear_solver import LinearSolver, LinearSolverError


class LUSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix, symmetric=False) -> None:
        super().__init__(mat, symmetric=symmetric)

        self.mat = mat
        try:
            self.solver = sp.sparse.linalg.splu(mat)
        except RuntimeError as err:
            logger.warn("LU decomposition failed: %s", err)
            raise LinearSolverError("LU decomposition failed")

    def solve(self, rhs, trans=False, initial_sol=None):
        trans_str = "T" if trans else "N"
        return self.solver.solve(rhs, trans=trans_str)
