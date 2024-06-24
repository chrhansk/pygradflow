from typing import Iterator, Optional

import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.newton import newton_method
from pygradflow.params import ActiveSetMethod, Params
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

    def compute_tau(self, initial_iterate: Iterate, rho: float) -> Optional[float]:
        problem = self.problem
        params = self.params
        active_set_method = params.active_set_method

        if active_set_method == ActiveSetMethod.Standard:
            return None
        elif active_set_method == ActiveSetMethod.SmallestActiveSet:
            return 0.0

        assert active_set_method == ActiveSetMethod.LargestActiveSet

        x = initial_iterate.x
        g = initial_iterate.aug_lag_deriv_x(rho)
        xl = problem.var_lb
        xu = problem.var_ub

        zero_g = np.isclose(g, 0.0)
        nonzero_g = np.logical_not(zero_g)
        pos_g = np.logical_and(g > 0.0, nonzero_g)
        neg_g = np.logical_and(g < 0.0, nonzero_g)

        tau_vals = np.zeros_like(x)

        tau_vals[pos_g] = (x[pos_g] - xl[pos_g]) / g[pos_g]
        tau_vals[neg_g] = (xu[neg_g] - x[neg_g]) / -g[neg_g]

        total_pos = np.sum(pos_g)
        total_neg = np.sum(neg_g)

        at_bounds = np.logical_or(np.isclose(x, xl), np.isclose(x, xu))

        zero_active = np.logical_and(zero_g, at_bounds)

        zero_active_size = np.sum(zero_active)

        logger.debug(
            "Range of active set size: %d to %d",
            zero_active_size,
            zero_active_size + total_pos + total_neg,
        )

        # return 1. / self.lamb
        return np.max(tau_vals)
