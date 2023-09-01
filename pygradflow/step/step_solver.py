import abc

from pygradflow.iterate import Iterate


class StepSolver(abc.ABC):
    def __init__(self, problem, params):
        self.problem = problem
        self.params = params
        self.n = problem.num_vars
        self.m = problem.num_cons

    def linear_solver(self, mat):
        from .linear_solver import linear_solver
        solver_type = self.params.linear_solver_type
        return linear_solver(mat, solver_type)

    @abc.abstractmethod
    def update_active_set(self, active_set):
        raise NotImplementedError

    def update_derivs(self, iterate: Iterate):
        raise NotImplementedError

    def solve(self, iterate: Iterate):
        raise NotImplementedError
