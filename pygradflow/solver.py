import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from pygradflow.cons_problem import ConstrainedProblem
from pygradflow.display import Format, problem_display
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.newton import newton_method
from pygradflow.params import Params
from pygradflow.penalty import penalty_strategy
from pygradflow.problem import Problem
from pygradflow.step.step_control import StepController, StepResult, step_controller


class SolverStatus(Enum):
    Optimal = auto()
    """
    The algorithm has converged to a solution satisfying
    the optimality conditions according to given tolerances
    """

    IterationLimit = auto()
    """
    Reached the iteration limit precribed by the algorithmic
    parameters
    """

    TimeLimit = auto()
    """
    Reached the time limit precribed by the algorithmic
    parameters
    """

    Unbounded = auto()
    """
    Problem appearst unbounded (found feasible point with extremely
    small objective value)
    """

    LocallyInfeasible = auto()
    """
    Local infeasibility detected (found infeasible point being
    a local minimum with respect to constraint violation)
    """

    @staticmethod
    def short_name(status):
        return {
            SolverStatus.Optimal: "optimal",
            SolverStatus.IterationLimit: "iteration_limit",
            SolverStatus.TimeLimit: "time_limit",
            SolverStatus.Unbounded: "unbounded",
            SolverStatus.LocallyInfeasible: "infeasible",
        }[status]

    @staticmethod
    def description(status):
        return {
            SolverStatus.Optimal: "Converged to first-order optimal solution",
            SolverStatus.IterationLimit: "Reached iteration limit",
            SolverStatus.TimeLimit: "Reached time limit",
            SolverStatus.Unbounded: "Problem appears unbounded",
            SolverStatus.LocallyInfeasible: "Local infeasibility detected",
        }[status]

    @staticmethod
    def success(status):
        """
        Returns
        -------
        bool
            Whether the status indicates a successful solve
        """
        return status == SolverStatus.Optimal


class SolverResult:
    """
    The result of a solution of a :py:class:`pygradflow.problem.Problem`
    instance with a :py:class:`pygradflow.solver.Solver`
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        status: SolverStatus,
        iterations: int,
        num_accepted_steps: int,
        total_time: float,
        dist_factor: float,
    ):
        self._x = x
        self._y = y
        self._d = d
        self._status = status
        self.iterations = iterations
        self.num_accepted_steps = num_accepted_steps
        self.total_time = total_time
        self.dist_factor = dist_factor

    @property
    def status(self) -> SolverStatus:
        """
        The status of the solve as a :py:class:`pygradflow.solver.SolverStatus`
        """
        return self._status

    @property
    def x(self) -> np.ndarray:
        """
        The primal solution :math:`x \\in \\mathbb{R}^{n}`
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        The dual solution :math:`y \\in \\mathbb{R}^{m}`
        """
        return self._y

    @property
    def d(self) -> np.ndarray:
        """
        The dual solution :math:`d \\in \\mathbb{R}^{n}`
        with respect to the variable bounds
        """
        return self._d

    def __repr__(self) -> str:
        return "SolverResult(status={0})".format(self.status)

    @property
    def success(self):
        return SolverStatus.success(self.status)


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

    def __init__(self, problem: Problem, params: Params = Params()) -> None:
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

        self.problem = ConstrainedProblem(problem)

        if params.validate_input:
            from .eval import ValidatingEvaluator

            self.evaluator = ValidatingEvaluator(self.problem, params)
        else:
            from .eval import SimpleEvaluator

            self.evaluator = SimpleEvaluator(self.problem, params)

        self.penalty = penalty_strategy(self.problem, params)
        self.rho = -1.0

    def compute_step(
        self, controller: StepController, iterate: Iterate, dt: float, display: bool
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

        return controller.compute_step(iterate, self.rho, dt, next_steps(), display)

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

        logger.info("%20s: %45e", "Bound violation", iterate.bound_violation)
        logger.info("%20s: %45e", "Constraint violation", iterate.cons_violation)
        logger.info("%20s: %45e", "Dual violation", iterate.stat_res)

    def solve(
        self, x0: Optional[np.ndarray] = None, y0: Optional[np.ndarray] = None
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
        problem = self.problem
        params = self.params
        dtype = params.dtype

        display = problem_display(problem, params)

        n = problem.num_vars
        m = problem.num_cons

        orig_problem = self.orig_problem

        if x0 is None:
            orig_n = orig_problem.num_vars
            x0 = np.zeros((orig_n,), dtype=dtype)
            x0 = np.clip(x0, orig_problem.var_lb, orig_problem.var_ub)

        if y0 is None:
            orig_m = orig_problem.num_cons
            y0 = np.zeros((orig_m,), dtype=dtype)

        (x0, y0) = problem.transform_sol(x0, y0)

        x = x0.astype(dtype)
        y = y0.astype(dtype)

        assert (n,) == x.shape
        assert (m,) == y.shape

        logger.info("Solving problem with {0} variables, {1} constraints".format(n, m))

        lamb = params.lamb_init

        controller = step_controller(problem, params)

        self._deriv_check(x, y)

        iterate = Iterate(problem, params, x, y, self.evaluator)
        self.rho = self.penalty.initial(iterate)

        logger.debug("Initial Aug Lag: %.10e", iterate.aug_lag(self.rho))

        status = None
        start_time = time.time()
        last_time = start_time
        line_diff = 0
        iteration = 0

        logger.info(display.header)

        path_dist = 0.0
        initial_iterate = iterate
        accepted_steps = 0
        iteration = 0

        last_active_set = None
        last_display_iteration = -1

        while True:
            if line_diff == header_interval:
                line_diff = 0
                logger.info(display.header)

            if iterate.total_res <= params.opt_tol:
                logger.debug("Convergence achieved")
                status = SolverStatus.Optimal
                break

            if iterate.locally_infeasible(params.opt_tol, params.local_infeas_tol):
                logger.debug("Local infeasibility detected")
                status = SolverStatus.LocallyInfeasible
                break

            if (iterate.obj <= params.obj_lower_limit) and (
                iterate.is_feasible(params.opt_tol)
            ):
                logger.debug("Unboundedness detected")
                status = SolverStatus.Unbounded
                break

            curr_time = time.time()
            display_iterate = curr_time - last_time >= params.display_interval

            step_result = self.compute_step(
                controller, iterate, 1.0 / lamb, display_iterate
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

            primal_step_norm = np.linalg.norm(next_iterate.x - iterate.x)
            dual_step_norm = np.linalg.norm(next_iterate.y - iterate.y)

            if curr_time - start_time >= params.time_limit:
                logger.debug("Reached time limit")
                status = SolverStatus.TimeLimit
                break

            if display_iterate:
                last_time = curr_time
                line_diff += 1

                state = dict()
                state["iterate"] = iterate

                def compute_last_active_set():
                    if last_display_iteration + 1 == iteration:
                        return last_active_set
                    return None

                state["last_active_set"] = compute_last_active_set
                state["curr_active_set"] = lambda: step_result.active_set

                state["aug_lag"] = lambda: iterate.aug_lag(self.rho)
                state["iter"] = lambda: iteration + 1
                state["primal_step_norm"] = lambda: primal_step_norm
                state["dual_step_norm"] = lambda: dual_step_norm
                state["lamb"] = lambda: lamb
                state["step_accept"] = lambda: accept
                state["rcond"] = lambda: step_result.rcond

                logger.info(display.row(state))
                last_display_iteration = iteration

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

            if (params.iteration_limit is not None) and (
                iteration >= params.iteration_limit
            ):
                status = SolverStatus.IterationLimit
                logger.debug("Iteration limit reached")
                break

        curr_time = time.time()
        total_time = curr_time - start_time

        direct_dist = iterate.dist(initial_iterate)

        assert path_dist >= direct_dist

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
        d = iterate.bound_duals

        (x, y, d) = problem.restore_sol(x, y, d)

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
