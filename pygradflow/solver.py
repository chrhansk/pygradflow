from typing import List, Optional, Tuple

import numpy as np

from pygradflow.callbacks import Callbacks, CallbackType
from pygradflow.display import Format, StateData, print_problem_stats, solver_display
from pygradflow.eval import EvalError
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, PenaltyUpdate
from pygradflow.penalty import penalty_strategy
from pygradflow.problem import Problem
from pygradflow.result import SolverResult
from pygradflow.status import SolverStatus
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    step_controller,
)
from pygradflow.timer import Timer
from pygradflow.transform import Transformation

header_interval = 25


class Solver:
    """
    Solves a :py:class:`pygradflow.problem.Problem` instance according
    to the given :py:class:`pygradflow.params.Params`.
    The solver attempts to find a solution given by
    vectors :math:`x \\in \\mathbb{R}^{n}`,
    :math:`y \\in \\mathbb{R}^{m}`, and :math:`d \\in \\mathbb{R}^{n}`,
    approximately satisfying the stationarity
     .. math::
        \\begin{align}
            \\nabla_x f(x) + y^{T} J_c(x) + d = 0,
        \\end{align}

    feasibility

     .. math::
        \\begin{align}
            \\quad & l \\leq c(x) \\leq u \\\\
            \\quad & l^x \\leq x \\leq u^x \\\\
        \\end{align}

    and complementarity

     .. math::
        \\begin{align}
            y_i
            \\begin{cases}
                \\geq 0 & \\text{ if } c(x)_i = u_i \\\\
                \\leq 0 & \\text{ if } c(x)_i = l_i \\\\
                = 0 & \\text{ otherwise }
            \\end{cases} \\\\
            d_j
            \\begin{cases}
                \\geq 0 & \\text{ if } x_j = u^x_j \\\\
                \\leq 0 & \\text{ if } x_j = l^x_j \\\\
                = 0 & \\text{ otherwise }
            \\end{cases}
        \\end{align}

    conditions.
    """

    def __init__(
        self,
        problem: Problem,
        params: Params = Params(),
    ) -> None:
        """
        Creates a new solver

        Parameters
        ----------
        problem: pygradflow.problem.Problem
            The problem to be solved
        params: pygradflow.params.Params
            Parameters used by the solver
        """
        self.orig_problem = problem
        self.params = params
        self.callbacks = Callbacks()

        self.transform = Transformation(self.orig_problem, self.params)
        self.problem = self.transform.trans_problem

    def _compute_step(
        self,
        controller: StepController,
        iterate: Iterate,
        rho: float,
        dt: float,
        display: bool,
        timer: Timer,
    ) -> StepControlResult:

        assert rho != -1.0
        return controller.compute_step(iterate, rho, dt, display, timer)

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
                params,
            )

        logger.info("Finished derivative check")

    def print_result(
        self,
        total_time: float,
        status: SolverStatus,
        iterate: Iterate,
        iterations: int,
        accepted_steps: int,
        dist_factor: float,
        rho_init: float,
        rho_final: float,
        num_penalty_changes: float,
    ) -> None:
        rho = self.rho

        desc = "{:>45s}".format(SolverStatus.description(status))

        status_desc = Format.redgreen(desc, SolverStatus.success(status), bold=True)
        status_name = Format.bold("{:>20s}".format("Status"))

        logger.info("%20s: %45s", status_name, status_desc)
        logger.info("%20s: %45s", "Time", f"{total_time:.2f}s")
        logger.info("%20s: %45d", "Iterations", iterations)
        logger.info("%20s: %45d", "Accepted steps", accepted_steps)

        logger.info("%20s: %45e", "Distance factor", dist_factor)

        if self.params.penalty_update != PenaltyUpdate.Constant:
            logger.info("%20s: %45e", "Initial penalty", rho_init)
            logger.info("%20s: %45e", "Final penalty", rho_final)
            logger.info("%20s: %45d", "Penalty changes", num_penalty_changes)

        logger.info("%20s: %45e", "Objective", iterate.obj)
        logger.info("%20s: %45e", "Aug Lag violation", iterate.aug_lag_violation(rho))
        logger.info("%20s: %45e", "Aug Lag dual", iterate.aug_lag_dual())

        logger.info("%20s: %45e", "Constraint violation", iterate.cons_violation)
        logger.info("%20s: %45e", "Dual violation", iterate.stat_res)

        eval = self.evaluator

        eval_name = Format.bold("{:>20s}".format("Evaluations"))
        logger.info("%20s", eval_name)

        for component, num_evals in eval.num_evals.items():
            name = component.name()
            logger.info("%20s: %45d", name, num_evals)

    def _check_terminate(self, iterate, iteration, timer):
        params = self.params

        if (params.iteration_limit is not None) and (
            iteration >= params.iteration_limit
        ):
            logger.debug("Iteration limit reached")
            return SolverStatus.IterationLimit

        if timer.reached_time_limit():
            logger.debug("Reached time limit")
            return SolverStatus.TimeLimit

        if iterate.total_res <= params.opt_tol:
            logger.debug("Convergence achieved")
            return SolverStatus.Optimal

        if iterate.locally_infeasible(params.opt_tol, params.local_infeas_tol):
            logger.debug("Local infeasibility detected")
            return SolverStatus.LocallyInfeasible

        if (iterate.obj <= params.obj_lower_limit) and (
            iterate.is_feasible(params.opt_tol)
        ):
            logger.debug("Unboundedness detected")
            return SolverStatus.Unbounded

    def perform_iteration(
        self, x0: np.ndarray | float | None = None, y0: np.ndarray | float | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        params = self.params
        problem = self.problem
        iterate = self.transform.create_transformed_iterate(x0, y0)

        lamb = params.lamb_init
        rho_init = params.rho

        controller = step_controller(problem, params)

        timer = Timer(float(np.inf))

        step_result = self._compute_step(
            controller, iterate, rho_init, 1.0 / lamb, False, timer
        )

        iterate = step_result.iterate

        x = iterate.x
        y = iterate.y
        d = iterate.bounds_dual

        return self.transform.restore_sol(x, y, d)

    def solve(
        self,
        x0: np.ndarray | float | None = None,
        y0: np.ndarray | float | None = None,
    ) -> SolverResult:
        """
        Solves the problem starting from the given primal / dual point

        Parameters
        ----------
        x0: np.ndarray | float | None
            The initial primal point :math:`x_0 \\in \\mathbb{R}^{n}`
        y0: np.ndarray | float | None
            The initial dual point :math:`y_0 \\in \\mathbb{R}^{m}`

        Returns
        -------
        pygradflow.solver.SolverResult
            The result of the solving process, including primal and dual
            solutions
        """

        params = self.params
        problem = self.problem

        self.evaluator = self.transform.evaluator

        self.penalty_strategy = penalty_strategy(self.problem, params)
        self.rho = -1.0

        display = solver_display(problem, params)

        iterate = self.transform.create_transformed_iterate(x0, y0)

        try:
            iterate.check_eval()
        except EvalError as e:
            raise Exception("Failed to evaluate initial iterate") from e

        print_problem_stats(problem, iterate)

        lamb = params.lamb_init

        controller = step_controller(problem, params)

        self._deriv_check(iterate.x, iterate.y)

        rho_init = self.penalty_strategy.initial(iterate)
        self.rho = rho_init

        logger.debug("Initial Aug Lag: %.10e", iterate.aug_lag(self.rho))

        status = None
        iteration = 0

        logger.info(display.header)

        path_dist = 0.0
        initial_iterate = iterate
        accepted_steps = 0
        iteration = 0

        num_penalty_changes = 0

        timer = Timer(params.time_limit)

        if params.collect_path:
            path: Optional[List[np.ndarray]] = [initial_iterate.z]
            path_times = [0.0]
        else:
            path = None

        while True:
            status = self._check_terminate(iterate, iteration, timer)
            if status is not None:
                break

            display_iterate = display.should_display()

            step_result = self._compute_step(
                controller, iterate, self.rho, 1.0 / lamb, display_iterate, timer
            )

            x = iterate.x
            y = iterate.y

            next_iterate = step_result.iterate
            accept = step_result.accepted
            lamb = step_result.lamb

            if lamb >= params.lamb_max:
                raise Exception(
                    f"Inverse step size {lamb} exceeded maximum {params.lamb_max} (incorrect derivatives?)"
                )

            primal_step_norm = float(np.linalg.norm(next_iterate.x - iterate.x))
            dual_step_norm = float(np.linalg.norm(next_iterate.y - iterate.y))

            self.callbacks(CallbackType.ComputedStep, iterate, next_iterate, accept)

            if display_iterate:
                state = StateData(iterate, step_result)
                state["iterate"] = iterate
                state["active_set"] = lambda _, step_result: step_result.active_set
                state["obj_nonlin"] = lambda iterate, _: iterate.obj_nonlin(
                    step_result.iterate
                )

                if problem.num_cons > 0:
                    state["cons_nonlin"] = lambda iterate, _: np.linalg.norm(
                        iterate.cons_nonlin(step_result.iterate), ord=np.inf
                    )

                state["aug_lag"] = lambda iterate, _: iterate.aug_lag(self.rho)
                state["obj"] = lambda iterate, _: iterate.obj()
                state["iter"] = iteration + 1
                state["primal_step_norm"] = primal_step_norm
                state["dual_step_norm"] = dual_step_norm
                state["lamb"] = lamb
                state["step_accept"] = accept
                state["rcond"] = lambda _, step_result: step_result.rcond

                logger.info(display.row(state))

            if accept:
                penalty_result = self.penalty_strategy.update(iterate, next_iterate)
                next_rho = penalty_result.next_rho
                accept = penalty_result.accept

            if accept:

                if next_rho != self.rho:
                    logger.debug(
                        "Updating penalty parameter from %e to %e", self.rho, next_rho
                    )
                    self.rho = next_rho
                    num_penalty_changes += 1

                if path is not None:
                    path.append(next_iterate.z)
                    path_times.append(path_times[-1] + (1.0 / lamb))

                iterate = next_iterate

                path_dist += primal_step_norm + dual_step_norm
                accepted_steps += 1

            iteration += 1

        total_time = timer.elapsed()

        direct_dist = iterate.dist(initial_iterate)

        assert (path_dist >= direct_dist) or (np.isclose(path_dist, direct_dist))

        dist_factor = path_dist / direct_dist if direct_dist != 0.0 else 1.0
        iterations = iteration

        assert status is not None

        self.print_result(
            total_time=total_time,
            status=status,
            iterate=iterate,
            iterations=iterations,
            accepted_steps=accepted_steps,
            dist_factor=dist_factor,
            rho_init=rho_init,
            rho_final=self.rho,
            num_penalty_changes=num_penalty_changes,
        )

        x = iterate.x
        y = iterate.y
        d = iterate.bounds_dual

        (x, y, d) = self.transform.restore_sol(x, y, d)

        result = SolverResult(
            problem,
            x,
            y,
            d,
            status,
            iterations=iterations,
            num_accepted_steps=accepted_steps,
            total_time=total_time,
            dist_factor=dist_factor,
            final_scaled_obj=iterate.obj,
            final_stat_res=iterate.stat_res,
            final_cons_violation=iterate.cons_violation,
        )

        if path is not None:
            complete_path = np.vstack(path).T
            model_times = np.hstack(path_times)
            result._set_path(complete_path, model_times)

        return result
