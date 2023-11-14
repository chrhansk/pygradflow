import copy

import numpy as np

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.step.step_solver import StepSolver, StepResult
from pygradflow.iterate import Iterate
from pygradflow.params import Params
from pygradflow.problem import Problem


class StandardStepSolver(StepSolver):
    """
    Standard Newton step solver. Computes the step based on the
    value and derivative of the implicit function.
    """

    def __init__(
        self,
        problem: Problem,
        params: Params,
        orig_iterate: Iterate,
        dt: float,
        rho: float,
    ) -> None:
        super().__init__(problem, params)

        self.orig_iterate = orig_iterate
        self.func = ImplicitFunc(problem, orig_iterate, dt)

        self.active_set = None
        self.jac = None
        self.hess = None
        self.rho = rho

    def _compute_deriv(self) -> None:
        assert self.active_set is not None
        assert self.jac is not None
        assert self.hess is not None

        self.deriv = self.func.deriv(self.jac, self.hess, self.active_set)

    def _reset_deriv(self) -> None:
        self.deriv = None
        self.solver = None

    def update_derivs(self, iterate: Iterate) -> None:
        self.jac = copy.copy(iterate.aug_lag_deriv_xy())
        self.hess = copy.copy(iterate.aug_lag_deriv_xx(self.rho))
        self._reset_deriv()

    def update_active_set(self, active_set: np.ndarray) -> None:
        self.active_set = copy.copy(active_set)
        self._reset_deriv()

    def solve(self, iterate: Iterate) -> StepResult:
        if self.deriv is None:
            self._compute_deriv()

        if self.solver is None:
            self.solver = self.linear_solver(self.deriv)

        rhs = self.func.value_at(iterate, self.rho, self.active_set)

        s = self.solver.solve(rhs)

        n = self.n
        m = self.m

        assert s.shape == (n + m,)

        dx = s[:n]
        dy = s[n:]

        params = self.params

        rcond = None
        if params.report_rcond:
            rcond = self.estimate_rcond(self.deriv, self.solver)

        return StepResult(iterate, dx, dy, rcond)
