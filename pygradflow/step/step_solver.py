import abc

import numpy as np
import scipy as sp

from pygradflow.iterate import Iterate
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.linear_solver import LinearSolver


class StepSolver(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params
        self.n = problem.num_vars
        self.m = problem.num_cons

    def linear_solver(self, mat: sp.sparse.spmatrix) -> LinearSolver:
        from .linear_solver import linear_solver

        solver_type = self.params.linear_solver_type
        return linear_solver(mat, solver_type)

    @abc.abstractmethod
    def update_active_set(self, active_set: np.ndarray):
        raise NotImplementedError

    def update_derivs(self, iterate: Iterate):
        raise NotImplementedError

    def solve(self, iterate: Iterate):
        raise NotImplementedError
