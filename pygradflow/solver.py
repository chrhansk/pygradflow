import time
from typing import Optional

import numpy as np

from pygradflow.callbacks import Callbacks, CallbackType
from pygradflow.display import Format, StateData, solver_display
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.newton import newton_method
from pygradflow.params import Params
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

    def compute_step(
        self,
        controller: StepController,
        iterate: Iterate,
        dt: float,
        display: bool,
        timer: Timer,
    ) -> StepControlResult:
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

        return controller.compute_step(
            iterate, self.rho, dt, next_steps(), display, timer
        )

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

    def print_problem_stats(self, problem, iterate):
        num_vars = problem.num_vars
        num_cons = problem.num_cons

        logger.info(
            "Solving problem with %s variables, %s constraints", num_vars, num_cons
        )

        cons_jac = iterate.cons_jac
        lag_hess = iterate.lag_hess(iterate.y)

        var_lb = problem.var_lb
        var_ub = problem.var_ub

        inf_lb = var_lb == -np.inf
        inf_ub = var_ub == np.inf

        unbounded = np.logical_and(inf_lb, inf_ub)
        ranged = np.logical_and(np.logical_not(inf_lb), np.logical_not(inf_ub))
        bounded_below = np.logical_and(np.logical_not(inf_lb), inf_ub)
        bounded_above = np.logical_and(inf_lb, np.logical_not(inf_ub))
        bounded = np.logical_or(bounded_below, bounded_above)

        num_unbounded = np.sum(unbounded)
        num_ranged = np.sum(ranged)
        num_bounded = np.sum(bounded)

        has_cons = num_cons > 0

        logger.info("           Hessian nnz: %s", lag_hess.nnz)
        if has_cons:
            logger.info("          Jacobian nnz: %s", cons_jac.nnz)

        logger.info("        Free variables: %s", num_unbounded)
        logger.info("     Bounded variables: %s", num_bounded)
        logger.info("      Ranged variables: %s", num_ranged)

        if not has_cons:
            return

        cons_lb = problem.cons_lb
        cons_ub = problem.cons_ub

        equation = cons_lb == cons_ub
        ranged = np.logical_and(np.isfinite(cons_lb), np.isfinite(cons_ub))
        ranged = np.logical_and(ranged, np.logical_not(equation))

        num_equations = equation.sum()
        num_ranged = ranged.sum()
        num_inequalities = num_cons - num_equations - num_ranged

        logger.info("  Equality constraints: %s", num_equations)
        logger.info("Inequality constraints: %s", num_inequalities)
        logger.info("    Ranged constraints: %s", num_ranged)

    def solve(
        self,
        x0: Optional[np.ndarray] = None,
        y0: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """
        Solves the problem starting from the given primal / dual point

        Parameters
        ----------
        x0: np.ndarray
            The initial primal point :math:`x_0 \\in \\mathbb{R}^{n}`
        y0: np.ndarray
            The initial dual point :math:`y_0 \\in \\mathbb{R}^{m}`

        Returns
        -------
        pygradflow.solver.SolverResult
            The result of the solving process, including primal and dual
            solutions
        """

        self.transform = Transformation(self.orig_problem, self.params, x0, y0)

        self.problem = self.transform.trans_problem

        params = self.params
        problem = self.problem

        self.evaluator = self.transform.evaluator

        self.penalty = penalty_strategy(self.problem, params)
        self.rho = -1.0

        display = solver_display(problem, params)

        iterate = self.transform.initial_iterate

        self.print_problem_stats(problem, iterate)

        lamb = params.lamb_init

        controller = step_controller(problem, params)

        self._deriv_check(iterate.x, iterate.y)

        self.rho = self.penalty.initial(iterate)

        logger.debug("Initial Aug Lag: %.10e", iterate.aug_lag(self.rho))

        status = None
        iteration = 0

        logger.info(display.header)

        path_dist = 0.0
        initial_iterate = iterate
        accepted_steps = 0
        iteration = 0

        last_active_set = None
        last_display_iteration = -1

        timer = Timer(params.time_limit)

        while True:
            status = self._check_terminate(iterate, iteration, timer)
            if status is not None:
                break

            display_iterate = display.should_display()

            step_result = self.compute_step(
                controller, iterate, 1.0 / lamb, display_iterate, timer
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
                state = StateData()
                state["iterate"] = iterate
                state["active_set"] = lambda: step_result.active_set
                state["obj_nonlin"] = lambda: iterate.obj_nonlin(next_iterate)

                if problem.num_cons > 0:
                    state["cons_nonlin"] = lambda: np.linalg.norm(
                        iterate.cons_nonlin(next_iterate), ord=np.inf
                    )

                state["aug_lag"] = lambda: iterate.aug_lag(self.rho)
                state["obj"] = lambda: iterate.obj()
                state["iter"] = iteration + 1
                state["primal_step_norm"] = lambda: primal_step_norm
                state["dual_step_norm"] = lambda: dual_step_norm
                state["lamb"] = lambda: lamb
                state["step_accept"] = lambda: accept
                state["rcond"] = lambda: step_result.rcond

                logger.info(display.row(state))

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

                path_dist += primal_step_norm + dual_step_norm
                accepted_steps += 1

                if (lamb <= params.lamb_term) and (delta <= params.opt_tol):
                    logger.debug("Convergence achieved")
                    status = SolverStatus.Optimal
                    break

            iteration += 1
            last_active_set = step_result.active_set
            # last_active_set = iterate.active_set

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
        )

        x = iterate.x
        y = iterate.y
        d = iterate.bounds_dual

        (x, y, d) = self.transform.restore_sol(x, y, d)

        return SolverResult(
            x,
            y,
            d,
            status,
            iterations=iterations,
            num_accepted_steps=accepted_steps,
            total_time=total_time,
            dist_factor=dist_factor,
        )
