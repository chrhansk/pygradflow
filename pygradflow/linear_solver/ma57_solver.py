from typing import Optional

import numpy as np
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

        size = self.mat.shape[0]

        if size == 0:
            self.solver = None
            return

        def symbolic():
            return self.solver.do_symbolic_factorization(self.mat, raise_on_error=False)

        self._try_fact(symbolic)

        def numeric():
            return self.solver.do_numeric_factorization(self.mat, raise_on_error=False)

        self._try_fact(numeric)

    def _try_fact(self, func):
        num_tries = 10
        for _ in range(num_tries):
            status = func()

            if status.status == LinearSolverStatus.not_enough_memory:
                self.solver.increase_memory_allocation()
                continue

            self._handle_fact_status(status)
            break
        else:
            raise Exception("Failed to compute symbolic factorization")

    def _handle_fact_status(self, status):
        if status.status == LinearSolverStatus.successful:
            return

        if status.status == LinearSolverStatus.singular:
            raise LinearSolverError("Matrix is singular")

        raise Exception("Failed to compute factorization: %s", status)

    def solve(self, rhs, trans=False, initial_sol=None):
        from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus

        if self.solver is None:
            assert rhs.size == 0
            return np.zeros_like(rhs)

        assert not trans

        x, status = self.solver.do_back_solve(rhs)

        if status.status != LinearSolverStatus.successful:
            raise LinearSolverError("Failed to compute solution")

        return x

    def num_neg_eigvals(self):
        if self.solver is None:
            return 0
        return self.solver.get_info(24)

    def rcond(self) -> Optional[float]:
        # TODO: Find out how to get rcond from MA57
        # if self.report_rcond:
        #     return self.solver.get_info(27)

        return None
