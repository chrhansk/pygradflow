from abc import ABC, abstractmethod
from typing import Optional

import scipy as sp
from numpy import ndarray


class LinearSolverError(Exception):
    """
    Error signaling that the linear solver failed, e.g. because the
    matrix is (near) singular. The solver attempts to recover by
    reducing the step size
    """

    pass


class LinearSolver(ABC):

    def __init__(self, matrix: sp.sparse.spmatrix, symmetric=False):
        self.symmetric = symmetric

    @abstractmethod
    def solve(self, rhs: ndarray, trans: bool = False, initial_sol=False) -> ndarray:
        raise NotImplementedError()

    def num_neg_eigvals(self) -> Optional[int]:
        return None

    def rcond(self) -> Optional[float]:
        return None
