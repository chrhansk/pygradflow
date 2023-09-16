import scipy as sp

from pygradflow.params import LinearSolverType
from numpy import ndarray
from scipy.sparse._csc import csc_matrix


class LinearSolver:
    def solve(self, b: ndarray) -> ndarray:
        raise NotImplementedError


class MINRESSolver(LinearSolver):
    def __init__(self, mat):
        self.mat = mat

    def solve(self, b):
        return sp.sparse.linalg.minres(self.mat, b)[0]


class GMRESSolver(LinearSolver):
    def __init__(self, mat: csc_matrix) -> None:
        self.mat = mat

    def solve(self, b: ndarray) -> ndarray:
        return sp.sparse.linalg.gmres(self.mat, b, atol=0.0)[0]


def linear_solver(
    mat: sp.sparse.spmatrix, solver_type: LinearSolverType
) -> LinearSolver:
    if solver_type == LinearSolverType.LU:
        return sp.sparse.linalg.splu(mat)
    elif solver_type == LinearSolverType.MINRES:
        return MINRESSolver(mat)
    else:
        assert solver_type == LinearSolverType.GMRES

        return GMRESSolver(mat)
