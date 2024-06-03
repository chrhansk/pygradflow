import numpy as np

from pygradflow.controller import ControllerSettings, LogController
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.newton_control import NewtonController
from pygradflow.step.step_control import StepControlResult


class ResiduumRatioController(NewtonController):
    def __init__(self, problem: Problem, params: Params) -> None:
        settings = ControllerSettings.from_params(params)
        self.controller = LogController(settings, params.theta_ref)
        super().__init__(problem, params)

    def step(self, iterate, rho, dt, display, timer):
        assert dt > 0.0
        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        next_steps = self.newton_steps(iterate, rho, dt)

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
