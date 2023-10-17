import abc
from typing import Iterator

import numpy as np

from pygradflow.log import logger

from pygradflow.controller import ControllerSettings, LogController
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.params import Params
from pygradflow.problem import Problem


class StepResult:
    def __init__(self, iterate: Iterate, lamb: float, accepted: bool) -> None:
        self.iterate = iterate
        self.lamb = lamb
        self.accepted = accepted


class StepController(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params
        self.lamb = params.lamb_init

    @abc.abstractmethod
    def step(self, iterate, rho, dt, next_iterates):
        raise NotImplementedError()


class ExactController(StepController):
    def __init__(self, problem, params):
        super().__init__(problem, params)

    def step(self, iterate, rho, dt, next_iterates):
        assert dt > 0.0
        lamb = 1.0 / dt

        func = ImplicitFunc(self.problem, iterate, dt)

        def func_val(iterate):
            return np.linalg.norm(func.value_at(iterate, rho))

        cur_func_val = func_val(iterate)

        for i in range(10):
            next_iterate = next(next_iterates)

            next_func_val = func_val(next_iterate)
            logger.info(f"Func val: {next_func_val}")

            if next_func_val <= self.params.newton_tol:
                logger.debug("Newton method converged in %d iterations", i + 1)
                return StepResult(next_iterate, 0.5 * lamb, True)
            elif next_func_val > cur_func_val:
                break

        logger.debug("Newton method did not converge")

        return StepResult(next_iterate, 2.0 * lamb, False)


class DistanceRatioController(StepController):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)
        settings = ControllerSettings.from_params(params)
        self.controller = LogController(settings, params.theta_ref)

    def step(
        self, iterate: Iterate, rho: float, dt: float, next_iterates: Iterator[Iterate]
    ) -> StepResult:
        assert dt > 0.0
        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        func = ImplicitFunc(problem, iterate, dt)

        mid_iterate = next(next_iterates)

        if np.linalg.norm(func.value_at(mid_iterate, rho)) <= params.newton_tol:
            lamb_n = max(lamb * params.lamb_red, params.lamb_min)
            logger.info("Newton converged during first iteration, lamb_n = %f", lamb_n)
            return StepResult(mid_iterate, lamb_n, True)

        first_diff = mid_iterate.dist(iterate)

        if first_diff == 0.:
            return StepResult(mid_iterate, lamb, True)

        final_iterate = next(next_iterates)

        second_diff = final_iterate.dist(mid_iterate)

        if second_diff == 0.:
            return StepResult(final_iterate, lamb, True)

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
            "StepController: theta: %e, accepted: %e, lamb: %e, lamb_n: %e",
            theta,
            accepted,
            lamb,
            lamb_n,
        )

        self.lamb = lamb_n

        return StepResult(final_iterate, lamb_n, accepted)
