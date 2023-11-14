import scipy as sp

from pygradflow.params import LinearSolverType
from numpy import ndarray


class LinearSolver:
    def solve(self, b: ndarray, trans: bool = False) -> ndarray:
        raise NotImplementedError


class MINRESSolver(LinearSolver):
    def __init__(self, mat):
        self.mat = mat

    def solve(self, b, trans=False):
        # matrix should be symmetric anyways
        return sp.sparse.linalg.minres(self.mat, b)[0]


class GMRESSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat

    def solve(self, b: ndarray, trans=False) -> ndarray:
        mat = self.mat.T if trans else self.mat
        return sp.sparse.linalg.gmres(mat, b, atol=0.0)[0]


class LUSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat
        self.solver = sp.sparse.linalg.splu(mat)

    def solve(self, b: ndarray, trans=False) -> ndarray:
        trans_str = 'T' if trans else 'N'
        return self.solver.solve(b, trans=trans_str)


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
