import time
from enum import Enum, auto

import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.newton import newton_method
from pygradflow.penalty import penalty_strategy

from pygradflow.step.step_control import (
    StepResult,
    step_controller,
    StepController,
)

from pygradflow.display import problem_display


class SolverStatus(Enum):
    Converged = auto(),
    IterationLimit = auto(),
    TimeLimit = auto()


class SolverResult:
    def __init__(self, x, y, d, success, status):
        self.x = x
        self.y = y
        self.d = d
        self.success = success
        self.status = status


header_interval = 25


class Solver:
    def __init__(self, problem: Problem, params: Params = Params()) -> None:

        self.problem = problem
        self.params = params

        if params.validate_input:
            from .eval import SimpleEvaluator
            self.evaluator = SimpleEvaluator(problem, params)
        else:
            from .eval import ValidatingEvaluator
            self.evaluator = ValidatingEvaluator(problem, params)

        self.penalty = penalty_strategy(problem, params)
        self.rho = -1.0

    def compute_step(
        self, controller: StepController, iterate: Iterate, dt: float
    ) -> StepResult:
        problem = self.problem
        params = self.params
        assert self.rho != -1.0

        method = newton_method(problem, params, iterate, dt, self.rho)

        def next_steps():
            curr_iterate = iterate
            while True:
                next_step = method.step(curr_iterate)
                yield next_step
                curr_iterate = next_step.iterate

        return controller.step(iterate, self.rho, dt, next_steps())

    def _deriv_check(self, x: np.ndarray, y: np.ndarray) -> None:
        from pygradflow.deriv_check import deriv_check
        from pygradflow.params import DerivCheck

        eval = self.evaluator
        params = self.params
        deriv_check_type = params.deriv_check

        if deriv_check_type == DerivCheck.NoCheck:
            return

        if deriv_check_type & DerivCheck.CheckFirst:
            logger.info("Checking objective derivative")
            deriv_check(lambda x: eval.obj(x), x, eval.obj_grad(x), params)

            logger.info("Checking constraint derivative")
            deriv_check(lambda x: eval.cons(x), x, eval.cons_jac(x), params)

        if deriv_check_type & DerivCheck.CheckSecond:
            logger.info("Checking Hessian")

            deriv_check(
                lambda x: eval.obj_grad(x) + eval.cons_jac(x).T.dot(y),
                x,
                eval.lag_hess(x, y),
                params)

    def print_result(self, iterate: Iterate) -> None:
        rho = self.rho

        logger.info("%30s: %14e", "Objective", iterate.obj)
        logger.info("%30s: %14e", "Aug Lag violation", iterate.aug_lag_violation(rho))
        logger.info("%30s: %14e", "Aug Lag dual", iterate.aug_lag_dual())

        logger.info("%30s: %14e", "Bound violation", iterate.bound_violation)
        logger.info("%30s: %14e", "Constraint violation", iterate.cons_violation)
        logger.info("%30s: %14e", "Dual violation", iterate.stat_res)

    def solve(self, x_0: np.ndarray, y_0: np.ndarray) -> SolverResult:
        problem = self.problem
        params = self.params
        dtype = params.dtype

        display = problem_display(problem)

        x = x_0.astype(dtype)
        y = y_0.astype(dtype)

        (n,) = x.shape
        (m,) = y.shape

        logger.info("Solving problem with {0} variables, {1} constraints".format(n, m))

        lamb = params.lamb_init

        success = True

        controller = step_controller(problem, params)

        self._deriv_check(x, y)

        iterate = Iterate(problem, params, x, y, self.evaluator)
        self.rho = self.penalty.initial(iterate)

        logger.info("Initial Aug Lag: %.10e", iterate.aug_lag(self.rho))

        status = None
        start_time = time.time()
        last_time = start_time
        line_diff = 0
        iteration = 0

        logger.info(display.header)

        for iteration in range(params.num_it):
            if line_diff == header_interval:
                line_diff = 0
                logger.info(display.header)

            if iterate.total_res <= params.opt_tol:
                logger.info("Convergence achieved")
                status = SolverStatus.Converged
                break

            step_result = self.compute_step(controller, iterate, 1.0 / lamb)

            x = iterate.x
            y = iterate.y

            next_iterate = step_result.iterate
            accept = step_result.accepted
            lamb = step_result.lamb

            if lamb >= params.lamb_max:
                raise Exception(f"Inverse step size {lamb} exceeded maximum {params.lamb_max} (incorrect derivatives?)")

            primal_step_norm = np.linalg.norm(next_iterate.x - iterate.x)
            dual_step_norm = np.linalg.norm(next_iterate.y - iterate.y)

            curr_time = time.time()

            if curr_time - start_time >= params.time_limit:
                logger.info("Reached time limit")
                status = SolverStatus.TimeLimit
                break

            if curr_time - last_time >= params.display_interval:
                last_time = curr_time
                line_diff += 1

                state = dict()
                state["iterate"] = iterate

                state["aug_lag"] = lambda: iterate.aug_lag(self.rho)
                state["iter"] = lambda: iteration + 1
                state["primal_step_norm"] = lambda: primal_step_norm
                state["dual_step_norm"] = lambda: dual_step_norm
                state["lamb"] = lambda: lamb
                state["step_accept"] = lambda: accept

                logger.info(display.row(state))

            if accept:
                # Accept
                next_rho = self.penalty.update(iterate, next_iterate)

                if next_rho != self.rho:
                    logger.debug("Updating penalty parameter from %e to %e",
                                 self.rho,
                                 next_rho)
                    self.rho = next_rho

                delta = iterate.dist(next_iterate)

                iterate = next_iterate

                if (lamb <= params.lamb_term) and (delta <= params.opt_tol):
                    logger.info("Convergence achieved")
                    status = SolverStatus.Converged
                    break

        else:
            success = False
            status = SolverStatus.IterationLimit
            logger.info("Iteration limit reached")

        self.print_result(iterate)

        x = iterate.x
        y = iterate.y
        d = iterate.bound_duals

        assert status is not None

        return SolverResult(x, y, d, success, status)
