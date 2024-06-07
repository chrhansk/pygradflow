from abc import ABC, abstractmethod

from pygradflow.solver import Solver


class Instance(ABC):
    def __init__(self, name, num_vars, num_cons):
        self.name = name
        self.num_vars = num_vars
        self.num_cons = num_cons

    @property
    def size(self):
        return self.num_vars + self.num_cons

    def solve(self, params):
        problem = self.problem()
        solver = Solver(problem, params)
        return solver.solve(self.x0(), self.y0())

    @abstractmethod
    def problem(self):
        raise NotImplementedError()

    @abstractmethod
    def x0(self):
        raise NotImplementedError()

    def y0(self):
        return 0.0
