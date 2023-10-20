from collections import namedtuple

import numpy as np
from termcolor import colored

from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, PenaltyUpdate
from pygradflow.problem import Problem
from pygradflow.newton import newton_method
from pygradflow.penalty import (
    ConstantPenalty,
    DualEquilibration,
    DualNormUpdate,
    PenaltyStrategy,
)
from pygradflow.step.step_control import (
    StepResult,
    DistanceRatioController,
    StepController,
)


Result = namedtuple("Result", ["x", "y", "d", "success"])


def bold(s: str) -> str:
    return colored(s, attrs=["bold"])


def print_header() -> None:
    header = "{0:^4} {1:^16} {2:^16} {3:^16} {4:^16} {5:^16} {6:^16} {7:^16} {8:^8}".format(
        "Iter",
        "Aug Lag",
        "Bound inf",
        "Cons inf",
        "Dual inf",
        "Primal step",
        "Dual step",
        "Lambda",
        "Type",
    )

    logger.info(bold(header))


def penalty_strategy(problem: Problem, params: Params) -> PenaltyStrategy:
    penalty_update = params.penalty_update

    if penalty_update == PenaltyUpdate.Constant:
        return ConstantPenalty(problem, params)
    elif penalty_update == PenaltyUpdate.DualNorm:
        return DualNormUpdate(problem, params)
    elif penalty_update == PenaltyUpdate.DualEquilibration:
        return DualEquilibration(problem, params)

    raise ValueError("Invalid penalty update strategy")


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

        def next_iterates():
            curr_iterate = iterate
            while True:
                next_iterate = method.step(curr_iterate)
                yield next_iterate
                curr_iterate = next_iterate

        return controller.step(iterate, self.rho, dt, next_iterates())

    def _deriv_check(self, x: np.ndarray, y: np.ndarray) -> None:
        from pygradflow.deriv_check import deriv_check

        eval = self.evaluator
        params = self.params

        logger.info("Checking objective derivative")

        deriv_check(lambda x: eval.obj(x), x, eval.obj_grad(x), params)

        logger.info("Checking constraint derivative")

        deriv_check(lambda x: eval.cons(x), x, eval.cons_jac(x), params)

        logger.info("Checking Hessian")

        deriv_check(
            lambda x: eval.obj_grad(x) + eval.cons_jac(x).T.dot(y),
            x,
            eval.lag_hess(x, y),
            params,
        )

    def print_result(self, iterate: Iterate) -> None:
        rho = self.rho

        logger.info("%30s: %.10e", "Objective", iterate.obj)
        logger.info("%30s: %.10e", "Aug Lag violation", iterate.aug_lag_violation(rho))
        logger.info("%30s: %.10e", "Aug Lag dual", iterate.aug_lag_dual())

        logger.info("%30s: %.10e", "Bound violation", iterate.bound_violation)
        logger.info("%30s: %.10e", "Constraint violation", iterate.cons_violation)
        logger.info("%30s: %.10e", "Dual violation", iterate.stat_res)

    def solve(self, x_0: np.ndarray, y_0: np.ndarray) -> Result:
        problem = self.problem
        params = self.params
        dtype = params.dtype

        x = np.copy(x_0).astype(dtype)
        y = np.copy(y_0).astype(dtype)

        (n,) = x.shape
        (m,) = y.shape

        logger.info("Solving problem with {0} variables, {1} constraints".format(n, m))

        lamb = params.lamb_init

        success = True

        controller = DistanceRatioController(problem, params)

        if params.deriv_check:
            self._deriv_check(x, y)

        iterate = Iterate(problem, params, x, y, self.evaluator)
        self.rho = self.penalty.initial(iterate)

        logger.info("Initial Aug Lag: %.10e", iterate.aug_lag(self.rho))

        for i in range(params.num_it):
            if (i % 25) == 0:
                print_header()

            if iterate.total_res <= params.opt_tol:
                logger.info("Convergence achieved")
                break

            step_result = self.compute_step(controller, iterate, 1.0 / lamb)

            x = iterate.x
            y = iterate.y

            next_iterate = step_result.iterate
            accept = step_result.accepted
            lamb = step_result.lamb

            accept_str = (
                colored("Accept", "green") if accept else colored("Reject", "red")
            )

            primal_step_norm = np.linalg.norm(next_iterate.x - iterate.x)
            dual_step_norm = np.linalg.norm(next_iterate.y - iterate.y)

            logger.info(
                "{0} {1:16.9e} {2:16e} {3:16e} {4:16e} {5:16e} {6:16e} {7:16e} {8:^8}".format(
                    bold("{0:4d}".format(i)),
                    iterate.aug_lag(self.rho),
                    iterate.bound_violation,
                    iterate.cons_violation,
                    iterate.stat_res,
                    primal_step_norm,
                    dual_step_norm,
                    lamb,
                    accept_str
                )
            )

            if accept:
                # Accept
                next_rho = self.penalty.update(iterate, next_iterate)

                if next_rho != self.rho:
                    logger.debug(
                        "Updating penalty parameter from %e to %e", self.rho, next_rho
                    )
                    self.rho = next_rho

                delta = iterate.dist(next_iterate)

                iterate = next_iterate

                if (lamb <= params.lamb_term) and (delta <= params.opt_tol):
                    logger.info("Convergence achieved")
                    break

        else:
            success = False
            logger.info("Iteration limit reached")

        self.print_result(iterate)

        x = iterate.x
        y = iterate.y
        d = iterate.bound_duals

        return Result(x, y, d, success)
