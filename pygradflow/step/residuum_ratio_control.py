import numpy as np

from pygradflow.controller import ControllerSettings, LogController
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.newton_control import NewtonController
from pygradflow.step.solver.step_solver import StepResult
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

        mid_value = func.value_at(mid_iterate, rho)
        mid_norm = np.linalg.norm(mid_value)

        self.display_step(0, mid_step)

        if mid_norm <= params.newton_tol:
            lamb_n = max(lamb * params.lamb_red, params.lamb_min)
            logger.debug("Newton converged during first iteration, lamb_n = %f", lamb_n)
            return StepControlResult.from_step_result(mid_step, lamb_n, True)

        orig_norm = np.linalg.norm(func.value_at(iterate, rho))

        theta = mid_norm / orig_norm
        accepted = theta <= params.theta_max

        logger.debug(
            "StepController: theta: %e, accepted: %s, lamb: %e",
            theta,
            accepted,
            lamb,
        )

        if accepted:
            lamb_mod = self.controller.update(theta)
            lamb_n = max(params.lamb_min, lamb / lamb_mod)
            self.lamb = lamb_n
            return StepControlResult.from_step_result(mid_step, lamb_n, accepted)

        # Step would be rejected
        lamb_n = lamb * params.lamb_inc

        if self.controller.error_sum > 0.0:
            self.controller.reset()

        # Recovery starts here

        active_set = func.compute_active_set(iterate, rho)
        mid_active_set = func.compute_active_set(mid_iterate, rho)

        if (active_set == mid_active_set).all():
            return StepControlResult.from_step_result(mid_step, lamb_n, accepted)

        mid_primal_value = mid_value[: self.problem.num_vars]

        mid_primal_norm = np.linalg.norm(mid_primal_value)

        mid_unchanged_primal_value = mid_primal_value[mid_active_set == active_set]
        mid_unchanged_primal_norm = np.linalg.norm(mid_unchanged_primal_value)

        norm_factor = mid_unchanged_primal_norm / mid_primal_norm

        if norm_factor >= 1e-6:
            self.lamb = lamb_n
            return StepControlResult.from_step_result(mid_step, lamb_n, accepted)

        dir = func.flow_rhs(mid_iterate, rho)

        # Track back until first active step change...

        primal_dir = dir[: self.problem.num_vars]
        dual_dir = dir[self.problem.num_vars :]

        pos_dir = primal_dir > 0.0
        neg_dir = primal_dir < 0.0

        lb = problem.var_lb
        ub = problem.var_ub

        pos_ratio = (ub[pos_dir] - mid_primal_value[pos_dir]) / primal_dir[pos_dir]
        neg_ratio = (lb[neg_dir] - mid_primal_value[neg_dir]) / primal_dir[neg_dir]

        recovery_dt = np.minimum(np.min(pos_ratio), np.min(neg_ratio))
        recovery_dt = np.minimum(dt, recovery_dt)

        recovery_dx = recovery_dt * primal_dir
        recovery_dy = recovery_dt * dual_dir

        recovery_step = StepResult(iterate, recovery_dx, recovery_dy, active_set)

        recovery_func = ImplicitFunc(problem, iterate, recovery_dt)

        recovery_value = recovery_func.value_at(recovery_step.iterate, rho)
        recovery_norm = np.linalg.norm(recovery_value)

        recovery_theta = recovery_norm / orig_norm
        accepted = recovery_theta <= params.theta_max

        if accepted:
            lamb_n = lamb
        else:
            lamb_n = 2.0 * lamb

        return StepControlResult.from_step_result(recovery_step, lamb_n, accepted=accepted)
