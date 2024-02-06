import abc
import logging
from typing import Iterator, Optional

import numpy as np

from pygradflow.controller import ControllerSettings, LogController
from pygradflow.display import inner_display
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, StepControlType
from pygradflow.problem import Problem
from pygradflow.step.linear_solver import LinearSolverError
from pygradflow.step.step_solver import StepResult


class StepControlResult:
    def __init__(
        self,
        iterate: Iterate,
        lamb: float,
        active_set,
        rcond: Optional[float],
        accepted: bool,
    ) -> None:
        self.iterate = iterate
        self.lamb = lamb
        self.active_set = active_set
        self.rcond = rcond
        self.accepted = accepted

    @staticmethod
    def from_step_result(
        step_result: StepResult, lamb: float, accepted: bool
    ) -> "StepControlResult":
        return StepControlResult(
            step_result.iterate,
            lamb,
            step_result.active_set,
            step_result.rcond,
            accepted,
        )


class StepController(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params
        self.lamb = params.lamb_init

    @abc.abstractmethod
    def step(
        self,
        iterate: Iterate,
        rho: float,
        dt: float,
        next_steps: Iterator[StepResult],
        display: bool,
    ) -> StepControlResult:
        raise NotImplementedError()

    def update_stepsize_after_fail(self, lamb) -> None:
        return 2.0 * lamb

    def compute_step(
        self,
        iterate: Iterate,
        rho: float,
        dt: float,
        next_steps: Iterator[StepResult],
        display: bool,
    ) -> StepControlResult:
        try:
            self.display = None
            if display:

                def res_func(next_iterate):
                    func = ImplicitFunc(self.problem, iterate, dt)
                    return np.linalg.norm(func.value_at(next_iterate, rho))

                self.res_func = res_func
                self.display = inner_display(self.problem, self.params)
                logger.debug("     %s", self.display.header)
            return self.step(iterate, rho, dt, next_steps, display)
        except LinearSolverError as e:
            logger.debug("Linear solver error during step computation: %s", e)
            lamb = 1.0 / dt
            lamb = self.update_stepsize_after_fail(lamb)
            return StepControlResult(iterate, lamb, None, None, False)

    def display_step(self, iteration, step):
        level = logger.getEffectiveLevel()
        if not self.display or level > logging.DEBUG:
            return

        iterate = step.iterate
        state = dict()
        state["iter"] = lambda: iteration
        state["residuum"] = lambda: self.res_func(iterate)
        state["dist"] = lambda: step.diff
        state["active_set_size"] = lambda: step.active_set.sum()
        logger.debug("     %s", self.display.row(state))


class ExactController(StepController):
    def __init__(self, problem, params, max_num_it=10, rate_bound=0.5):
        super().__init__(problem, params)
        self.max_num_it = max_num_it
        self.rate_bound = rate_bound

    def step(self, iterate, rho, dt, next_steps, display):
        assert dt > 0.0
        lamb = 1.0 / dt

        func = ImplicitFunc(self.problem, iterate, dt)

        def func_val(iterate):
            return np.linalg.norm(func.value_at(iterate, rho))

        curr_func_val = func_val(iterate)

        rcond = None
        active_set = None

        for i in range(self.max_num_it):
            next_step = next(next_steps)
            next_iterate = next_step.iterate
            active_set = next_step.active_set
            rcond = next_step.rcond

            self.display_step(i, next_step)

            next_func_val = func_val(next_iterate)
            logger.debug(f"Func val: {next_func_val}")

            if next_func_val <= self.params.newton_tol:
                logger.debug("Newton method converged in %d iterations", i + 1)
                return StepControlResult(
                    next_iterate, 0.5 * lamb, active_set, rcond, True
                )

            rate_est = next_func_val / curr_func_val

            if rate_est > self.rate_bound:
                logger.debug(
                    "Newton convergence rate (%f) exceeds allorw (%f)",
                    rate_est,
                    self.rate_bound,
                )
                break

            curr_func_val = next_func_val

        logger.debug("Newton method did not convergein %d iterations", self.max_num_it)

        return StepControlResult(next_iterate, 2.0 * lamb, active_set, rcond, False)


class ResiduumRatioController(StepController):
    def __init__(self, problem: Problem, params: Params) -> None:
        settings = ControllerSettings.from_params(params)
        self.controller = LogController(settings, params.theta_ref)
        super().__init__(problem, params)

    def step(self, iterate, rho, dt, next_steps, display):
        assert dt > 0.0
        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        func = ImplicitFunc(problem, iterate, dt)

        mid_step = next(next_steps)
        mid_iterate = mid_step.iterate

        mid_norm = np.linalg.norm(func.value_at(mid_iterate, rho))

        self.display_step(0, mid_step)

        if mid_norm <= params.newton_tol:
            lamb_n = max(lamb * params.lamb_red, params.lamb_min)
            logger.debug("Newton converged during first iteration, lamb_n = %f", lamb_n)
            return StepControlResult.from_step_result(mid_step, lamb, True)

        orig_norm = np.linalg.norm(func.value_at(iterate, rho))

        theta = mid_norm / orig_norm
        accepted = theta <= params.theta_max

        if accepted:
            lamb_mod = self.controller.update(theta)
            lamb_n = max(params.lamb_min, lamb / lamb_mod)
        else:
            lamb_n = lamb * params.lamb_inc
            if self.controller.error_sum > 0.0:
                self.controller.reset()

        logger.debug(
            "StepController: theta: %e, accepted: %s, lamb: %e, lamb_n: %e",
            theta,
            accepted,
            lamb,
            lamb_n,
        )

        self.lamb = lamb_n
        return StepControlResult.from_step_result(mid_step, lamb_n, accepted)


class DistanceRatioController(StepController):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)
        settings = ControllerSettings.from_params(params)
        self.controller = LogController(settings, params.theta_ref)

    def step(self, iterate, rho, dt, next_steps, display):
        assert dt > 0.0
        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        func = ImplicitFunc(problem, iterate, dt)

        mid_step = next(next_steps)
        mid_iterate = mid_step.iterate

        self.display_step(0, mid_step)

        mid_func_norm = np.linalg.norm(func.value_at(mid_iterate, rho))

        if mid_func_norm <= params.newton_tol:
            lamb_n = max(lamb * params.lamb_red, params.lamb_min)
            logger.debug("Newton converged during first iteration, lamb_n = %f", lamb_n)
            return StepControlResult.from_step_result(mid_step, lamb_n, True)

        first_diff = mid_step.diff

        if first_diff == 0.0:
            return StepControlResult.from_step_result(mid_step, lamb, True)

        final_step = next(next_steps)

        self.display_step(1, final_step)

        second_diff = final_step.diff

        if second_diff == 0.0:
            return StepControlResult.from_step_result(final_step, lamb, True)

        logger.debug("First distance: %e, second distance: %e", first_diff, second_diff)

        theta = second_diff / first_diff

        accepted = theta <= params.theta_max

        if accepted:
            lamb_mod = self.controller.update(theta)
            lamb_n = max(params.lamb_min, lamb / lamb_mod)
        else:
            lamb_n = lamb * params.lamb_inc
            if self.controller.error_sum > 0.0:
                self.controller.reset()

        logger.debug(
            "StepController: theta: %e, accepted: %s, lamb: %e, lamb_n: %e",
            theta,
            accepted,
            lamb,
            lamb_n,
        )

        self.lamb = lamb_n
        return StepControlResult.from_step_result(final_step, lamb_n, accepted)


def step_controller(problem: Problem, params: Params) -> StepController:
    step_control_type = params.step_control_type

    if step_control_type == StepControlType.Exact:
        return ExactController(problem, params)
    elif step_control_type == StepControlType.ResiduumRatio:
        return ResiduumRatioController(problem, params)
    else:
        assert step_control_type == StepControlType.DistanceRatio
        return DistanceRatioController(problem, params)
