from typing import Optional

from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma57_interface import MA57

from .linear_solver import LinearSolver, LinearSolverError


class MA57Solver(LinearSolver):
    def __init__(self, mat, symmetric=False, report_rcond=False):
        super().__init__(mat, symmetric=symmetric)

        self.mat = mat.tocoo()
        self.solver = MA57()
        self.report_rcond = report_rcond

        # if report_rcond:
        #     self.solver.set_icntl(10, 1)

        status = self.solver.do_symbolic_factorization(self.mat)

        if status.status != LinearSolverStatus.successful:
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

    def rcond(self) -> Optional[float]:
        # TODO: Find out how to get rcond from MA57
        # if self.report_rcond:
        #     return self.solver.get_info(27)

        return None
