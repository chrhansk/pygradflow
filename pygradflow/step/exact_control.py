import numpy as np

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.log import logger
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    StepSolverError,
)


class ExactController(StepController):
    def __init__(self, problem, params, max_num_it=10, rate_bound=0.5):
        super().__init__(problem, params)
        self.max_num_it = max_num_it
        self.rate_bound = rate_bound

    def step(self, iterate, rho, dt, next_steps, display, timer):
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

            if timer.reached_time_limit():
                raise StepSolverError("Time limit reached")

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
                    "Newton convergence rate (%f) exceeded bound (%f)",
                    rate_est,
                    self.rate_bound,
                )
                break

            curr_func_val = next_func_val

        logger.debug("Newton method did not converge in %d iterations", self.max_num_it)

        return StepControlResult(next_iterate, 2.0 * lamb, active_set, rcond, False)
