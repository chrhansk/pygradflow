import scipy as sp
from numpy import ndarray

from pygradflow.params import LinearSolverType


class LinearSolverError(Exception):
    pass


class LinearSolver:
    def solve(self, b: ndarray, trans: bool = False) -> ndarray:
        raise NotImplementedError


class MINRESSolver(LinearSolver):
    def __init__(self, mat):
        self.mat = mat

    def solve(self, rhs, trans=False, initial_sol=None):
        # matrix should be symmetric anyways
        if initial_sol is not None:
            initial_sol = initial_sol()

        result = sp.sparse.linalg.minres(self.mat, rhs, x0=initial_sol)

        (sol, info) = result

        if info < 0:
            raise LinearSolverError("MINRES failed with error code {}".format(info))

        return sol


class GMRESSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat

    def solve(self, rhs: ndarray, trans=False, initial_sol=None) -> ndarray:
        mat = self.mat.T if trans else self.mat

        if initial_sol is not None:
            initial_sol = initial_sol()

        result = sp.sparse.linalg.gmres(mat, rhs, x0=initial_sol)

        (sol, info) = result

        if info < 0:
            raise LinearSolverError("GMRES failed with error code {}".format(info))

        return sol


class LUSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat
        self.solver = sp.sparse.linalg.splu(mat)

    def solve(self, rhs: ndarray, trans=False, initial_sol=None) -> ndarray:
        trans_str = "T" if trans else "N"
        return self.solver.solve(rhs, trans=trans_str)


def linear_solver(
    mat: sp.sparse.spmatrix, solver_type: LinearSolverType
) -> LinearSolver:
    if solver_type == LinearSolverType.LU:
        return LUSolver(mat)
    elif solver_type == LinearSolverType.MINRES:
        return MINRESSolver(mat)
    else:
        assert solver_type == LinearSolverType.GMRES

        return GMRESSolver(mat)
