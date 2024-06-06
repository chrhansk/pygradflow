import copy

import numpy as np

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.linear_solver import LinearSolverError
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.step_solver_error import StepSolverError

from .step_solver import StepResult, StepSolver


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
        self._func = ImplicitFunc(problem, orig_iterate, dt)

        self.rho = rho

    @property
    def func(self) -> ImplicitFunc:
        return self._func

    def _compute_deriv(self) -> None:
        assert self.active_set is not None
        assert self.jac is not None
        assert self.hess is not None

        self.deriv = self.func.deriv(self.jac, self.hess, self.active_set)

    def _reset_deriv(self) -> None:
        self.deriv = None
        self.solver = None

    def update_derivs(self, iterate: Iterate) -> None:
        self._jac = copy.copy(iterate.aug_lag_deriv_xy())
        self._hess = copy.copy(iterate.aug_lag_deriv_xx(self.rho))
        self._reset_deriv()

    def update_active_set(self, active_set: np.ndarray) -> None:
        self._active_set = copy.copy(active_set)
        self._reset_deriv()

    def solve(self, iterate: Iterate) -> StepResult:
        if self.deriv is None:
            self._compute_deriv()

        rhs = self.func.value_at(iterate, self.rho, self.active_set)

        try:
            if self.solver is None:
                self.solver = self.linear_solver(self.deriv)

            sol = self.solver.solve(rhs)
        except LinearSolverError as e:
            raise StepSolverError from e

        n = self.n
        m = self.m

        assert sol.shape == (n + m,)

        dx = sol[:n]
        dy = sol[n:]

        params = self.params

        rcond = None

        if params.report_rcond:
            try:
                rcond = self.estimate_rcond(self.deriv, self.solver)
            except LinearSolverError:
                pass

        return StepResult(iterate, dx, dy, self.active_set, rcond)
