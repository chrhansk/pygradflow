from typing import Iterator, Optional

import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.newton import newton_method
from pygradflow.params import ActiveSetType, Params
from pygradflow.problem import Problem
from pygradflow.step.solver.step_solver import StepResult
from pygradflow.step.step_control import StepController


class NewtonController(StepController):
    """
    Step controller working by solving the implicit Euler
    equations using a semi-smooth Newton method
    """

    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def newton_steps(
        self,
        orig_iterate: Iterate,
        rho: float,
        dt: float,
    ) -> Iterator[StepResult]:
        problem = self.problem
        params = self.params

        tau = self.compute_tau(orig_iterate, rho)
        self.method = newton_method(problem, params, orig_iterate, dt, rho, tau)
        curr_iterate = orig_iterate

        while True:
            next_step = self.method.step(curr_iterate)
            yield next_step
            curr_iterate = next_step.iterate

    def tau_vals(self, initial_iterate: Iterate, rho: float):
        problem = self.problem

        x = initial_iterate.x
        g = initial_iterate.aug_lag_deriv_x(rho)
        xl = problem.var_lb
        xu = problem.var_ub

        zero_g = np.isclose(g, 0.0)
        nonzero_g = np.logical_not(zero_g)
        pos_g = np.logical_and(g > 0.0, nonzero_g)
        neg_g = np.logical_and(g < 0.0, nonzero_g)

        tau_vals = np.full_like(x, fill_value=-1)

        tau_vals[pos_g] = (x[pos_g] - xl[pos_g]) / g[pos_g]
        tau_vals[neg_g] = (xu[neg_g] - x[neg_g]) / -g[neg_g]

        return tau_vals

    def compute_tau(self, initial_iterate: Iterate, rho: float) -> Optional[float]:
        params = self.params

        active_set_method = params.active_set_method

        if active_set_method is not None:
            return active_set_method(initial_iterate, self.lamb, rho)

        active_set_type = params.active_set_type

        if active_set_type == ActiveSetType.Standard:
            return None

        tau_vals = self.tau_vals(initial_iterate, rho)

        if active_set_type == ActiveSetType.SmallestActiveSet:
            if (tau_vals <= 0).all():
                return 1.0

            min_tau = np.min(tau_vals[tau_vals > 0])
            assert min_tau >= 0
            return 0.5 * min_tau

        return max(np.max(tau_vals), 1.0)
