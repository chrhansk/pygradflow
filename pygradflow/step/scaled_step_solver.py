import copy
from typing import Tuple, Optional

import numpy as np

from pygradflow.implicit_func import ScaledImplicitFunc
from pygradflow.step.step_solver import StepSolver, StepResult
from pygradflow.iterate import Iterate
from pygradflow.params import Params
from pygradflow.problem import Problem


class ScaledStepSolver(StepSolver):
    def __init__(
        self,
        problem: Problem,
        params: Params,
        orig_iterate: Iterate,
        dt: float,
        rho: float,
    ) -> None:
        super().__init__(problem, params)

        self.problem = problem
        self.orig_iterate = orig_iterate
        self.func = ScaledImplicitFunc(problem, orig_iterate, dt)

        self.dt = dt
        self.rho = rho

    def initial_rhs(
        self, iterate: Iterate
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        m = self.m

        rhs = self.func.value_at(iterate, self.rho, self.active_set)

        assert rhs.shape == (n + m,)

        rx = rhs[:n]
        ry = rhs[n:]

        active_indices = np.where(self.active_set)[0]
        inactive_indices = np.where(np.logical_not(self.active_set))[0]

        dt = self.dt

        b0 = dt * rx[active_indices]
        b1 = rx[inactive_indices]
        b2 = ry

        return (b0, b1, b2)

    def solve_scaled(self, b0, b1, b2t) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        raise NotImplementedError()

    def reset_deriv(self) -> None:
        self.deriv = None
        self.solver = None

    def update_derivs(self, iterate: Iterate) -> None:
        self.jac = copy.copy(iterate.aug_lag_deriv_xy())
        self.hess = copy.copy(iterate.aug_lag_deriv_xx(rho=0.0))
        self.reset_deriv()

    def update_active_set(self, active_set: np.ndarray) -> None:
        self.active_set = copy.copy(active_set)
        self.reset_deriv()

    def solve(self, iterate: Iterate) -> StepResult:
        (b0, b1, b2) = self.initial_rhs(iterate)

        n = self.n
        m = self.m

        rho = self.rho
        lamb = 1.0 / self.dt
        fact = 1.0 / (1.0 + lamb * rho)

        assert fact > 0.0

        b2t = fact * b2

        (sx, sy, rcond) = self.solve_scaled(b0, b1, b2t)

        assert sx.shape == (n,)
        assert sy.shape == (m,)

        dx = sx
        dy = fact * (sy - rho * b2)

        return StepResult(iterate, dx, dy, rcond)
