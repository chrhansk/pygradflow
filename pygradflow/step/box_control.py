import cyipopt
import numpy as np
import scipy as sp
from scipy.optimize import Bounds, minimize

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    StepSolverError,
)


class BoxReducedProblem(cyipopt.Problem):
    """
    Solves the primal-dual proximally regularized box-reduced problem
     .. math::
        \\begin{align}
            \\min_{x \\in \\mathbb{R}^{n}, w \\in \\mathbb{R}^{n}} \\quad
            & f(x) + \\frac{\\rho}{2} \\|c(x)\\|^{2} + \\frac{\\lambda}{2} (\\| x - \\hat{x} \\|^{2} + \\| -\\frac{1}{\\lambda}x - \\hat{y} \\|^{2}) \\\\
            \\text{s.t.} \\quad & l^x \\leq x \\leq u^x
        \\end{align}

    whose optimum coincides with the implicit Euler step of the augmented Lagrangian
    of the original problem. The problem only consists of box constraints.
    """

    def __init__(self, step_controller, iterate, lamb, rho):
        self.iterate = iterate
        self.step_controller = step_controller
        self.lamb = lamb
        self.rho = rho
        self.problem = iterate.problem

        self.var_lb = self.problem.var_lb
        self.var_ub = self.problem.var_ub

        self.num_vars = self.problem.num_vars

        super(BoxReducedProblem, self).__init__(
            n=self.num_vars,
            m=0,
            lb=self.var_lb,
            ub=self.var_ub,
        )

    def objective(self, x):
        return self.step_controller.objective(self.iterate, x, self.lamb, self.rho)

    def gradient(self, x):
        return self.step_controller.gradient(self.iterate, x, self.lamb, self.rho)

    def constraint(self, x):
        return np.array([])

    def jacobian(self, x):
        return np.empty((0, self.num_vars))

    def hessianstructure(self):
        x0 = self.x0
        hess = self.step_controller.hessian(self.iterate, x0, self.lamb, self.rho)
        hess = hess.tocoo()
        rows = hess.row
        cols = hess.col

        hess_filter = rows >= cols

        return rows[hess_filter], cols[hess_filter]

    def hessian(self, x, lag, obj_factor):
        hess = self.step_controller.hessian(self.iterate, x, self.lamb, self.rho)
        hess = hess.tocoo()

        rows = hess.row
        cols = hess.col
        data = hess.data

        hess_filter = rows >= cols

        return data[hess_filter]

    def set_options(self, timer):
        import logging

        logging.getLogger("cyipopt").setLevel(logging.WARNING)
        self.add_option("print_level", 0)
        # self.add_option("derivative_test", "second-order")
        # self.add_option("hessian_approximation", "limited-memory")
        # self.add_option("max_iter", 10)

        remaining = timer.remaining()

        if remaining <= 0.0:
            raise StepSolverError("Time limit reached")
        elif np.isfinite(remaining):
            self.add_option("max_cpu_time", remaining)

    @property
    def x0(self):
        return self.iterate.x

    def solve(self, timer):
        self.set_options(timer)
        x0 = self.x0

        # Solve using Ipopt
        x, info = super().solve(x0)

        if info["status"] != 0:
            raise StepSolverError("Ipopt failed to solve the problem")

        return x


class BoxReducedController(StepController):

    def objective(self, iterate, x, lamb, rho):
        eval = iterate.eval
        xhat = iterate.x
        yhat = iterate.y

        obj = eval.obj(x)
        cons = eval.cons(x)

        dx = x - xhat
        dx_norm_sq = np.dot(dx, dx)

        w = -1 / lamb * cons
        dy = w - yhat
        dy_norm_sq = np.dot(dy, dy)

        curr_obj = obj + 0.5 * rho * np.dot(cons, cons)
        curr_obj += 0.5 * lamb * (dx_norm_sq + dy_norm_sq)

        return curr_obj

    def gradient(self, iterate, x, lamb, rho):
        eval = iterate.eval
        xhat = iterate.x
        yhat = iterate.y

        obj_grad = eval.obj_grad(x)
        cons = eval.cons(x)
        cons_jac = eval.cons_jac(x)

        dx = x - xhat
        cons_jac_factor = (rho + (1 / lamb)) * cons + yhat
        return obj_grad + lamb * dx + cons_jac.T.dot(cons_jac_factor)

    def hessian(self, iterate, x, lamb, rho):
        eval = iterate.eval
        (n,) = x.shape

        yhat = iterate.y

        cons = eval.cons(x)
        jac = eval.cons_jac(x)
        cons_factor = 1 / lamb + rho
        y = cons_factor * cons + yhat

        hess = eval.lag_hess(x, y)
        hess += sp.sparse.diags([lamb], shape=(n, n))
        hess += cons_factor * (jac.T @ jac)

        return hess

    def solve_step_scipy(self, iterate, rho, dt, timer):
        problem = self.problem

        x = iterate.x
        lamb = 1.0 / dt

        lb = problem.var_lb
        ub = problem.var_ub

        def opt_res(x):
            grad = gradient(x)

            at_lb = np.isclose(x, lb)
            at_ub = np.isclose(x, lb)

            grad[at_lb] = np.maximum(grad[at_lb], 0.0)
            grad[at_ub] = np.minimum(grad[at_ub], 0.0)

            return np.linalg.norm(grad, ord=np.inf)

        def objective(x):
            return self.objective(iterate, x, lamb, rho)

        def gradient(x):
            return self.gradient(iterate, x, lamb, rho)

        def callback(x):
            if timer.reached_time_limit():
                raise StopIteration()

        result = minimize(
            objective,
            x,
            jac=gradient,
            bounds=Bounds(lb, ub),
            method="TNC",
            options={"gtol": 1e-8, "ftol": 0.0},
            callback=callback,
        )

        if not result.success:
            raise StepSolverError("Scipy failed to solve the problem")

        return result.x

    def solve_step_ipopt(self, iterate, rho, dt, timer):
        lamb = 1.0 / dt
        reduced_problem = BoxReducedProblem(self, iterate, lamb, rho)
        return reduced_problem.solve(timer=timer)

    def solve_step_box(self, iterate, rho, dt, timer):
        from .box_solver import BoxSolverError, solve_box_constrained

        problem = self.problem
        lamb = 1.0 / dt

        def objective(x):
            return self.objective(iterate, x, lamb, rho)

        def gradient(x):
            return self.gradient(iterate, x, lamb, rho)

        def hessian(x):
            return self.hessian(iterate, x, lamb, rho)

        try:
            return solve_box_constrained(
                iterate.x, objective, gradient, hessian, problem.var_lb, problem.var_ub
            )
        except BoxSolverError as e:
            raise StepSolverError("Box-constrained solver failed to converge") from e

    def step(
        self, iterate, rho: float, dt: float, display: bool, timer
    ) -> StepControlResult:

        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        # x = self.solve_step_ipopt(iterate, rho, dt, timer)

        # Note: The minimize function shipped with scipy
        # do not consistently produce high-quality solutions,
        # causing the optimization of the overall problem to fail.
        # x = self.solve_step_scipy(iterate, rho, dt, timer)
        x = self.solve_step_box(iterate, rho, dt, timer)

        eval = iterate.eval
        cons = eval.cons(x)
        w = (-1 / lamb) * cons
        y = iterate.y - w

        next_iterate = Iterate(problem, params, x, y, iterate.eval)

        implicit_func = ImplicitFunc(problem, iterate, dt)

        value = implicit_func.value_at(next_iterate, rho)

        residuum = np.linalg.norm(value)

        if residuum <= 1e-6:
            return StepControlResult(
                next_iterate, 0.5 * lamb, active_set=None, rcond=None, accepted=True
            )

        return StepControlResult(
            next_iterate, 2.0 * lamb, active_set=None, rcond=None, accepted=False
        )
