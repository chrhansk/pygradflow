from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
import sksparse.cholmod
from numpy import ndarray

from pygradflow.log import logger
from pygradflow.params import LinearSolverType


class LinearSolverError(Exception):
    """
    Error signaling that the linear solver failed, e.g. because the
    matrix is (near) singular. The solver attempts to recover by
    reducing the step size
    """

    pass


class LinearSolver(ABC):
    @abstractmethod
    def solve(self, rhs: ndarray, trans: bool = False, initial_sol=False) -> ndarray:
        raise NotImplementedError()

    def num_neg_eigvals(self):
        return None


class MINRESSolver(LinearSolver):
    def __init__(self, mat):
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


class GMRESSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat

    def solve(self, rhs: ndarray, trans=False, initial_sol=None) -> ndarray:
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


class LUSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix) -> None:
        self.mat = mat
        try:
            self.solver = sp.sparse.linalg.splu(mat)
        except RuntimeError as err:
            logger.warn("LU decomposition failed: %s", err)
            raise LinearSolverError("LU decomposition failed")

    def solve(self, rhs: ndarray, trans=False, initial_sol=None) -> ndarray:
        trans_str = "T" if trans else "N"
        return self.solver.solve(rhs, trans=trans_str)


class CholeskySolver(LinearSolver):
    def __init__(self, mat):
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


class MA57Solver(LinearSolver):
    def __init__(self, mat):
        from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
        from pyomo.contrib.pynumero.linalg.ma57_interface import MA57

        self.mat = mat.tocoo()
        self.solver = MA57()
        status = self.solver.do_symbolic_factorization(self.mat)

        if status.status != LinearSolverStatus.successful:
            import pdb

            pdb.set_trace()
            raise LinearSolverError("Failed to compute symbolic factorization")

        status = self.solver.do_numeric_factorization(self.mat)

        if status.status != LinearSolverStatus.successful:
            raise LinearSolverError("Failed to compute numeric factorization")

    def solve(self, rhs, trans=False, initial_sol=None):
        from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus

        assert not trans

        x, status = self.solver.do_back_solve(rhs)

        if status.status != LinearSolverStatus.successful:
            raise LinearSolverError("Failed to compute solution")

        return x

    def num_neg_eigvals(self):
        return self.solver.get_info(24)


def linear_solver(
    mat: sp.sparse.spmatrix, solver_type: LinearSolverType
) -> LinearSolver:
    if solver_type == LinearSolverType.LU:
        return LUSolver(mat)
    elif solver_type == LinearSolverType.MINRES:
        return MINRESSolver(mat)
    elif solver_type == LinearSolverType.Cholesky:
        return CholeskySolver(mat)
    elif solver_type == LinearSolverType.MA57:
        return MA57Solver(mat)
    else:
        assert solver_type == LinearSolverType.GMRES

        return GMRESSolver(mat)
