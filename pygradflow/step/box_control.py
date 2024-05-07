import time

import cyipopt
import numpy as np
from scipy.optimize import Bounds, minimize

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    StepSolverError,
)

max_num_it = 10000
max_num_linesearch_it = 40
t_init = 1.0
beta = 0.5


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

    def set_options(self, timer):
        import logging

        logging.getLogger("cyipopt").setLevel(logging.WARNING)
        self.add_option("print_level", 0)
        # self.add_option("derivative_test", "first-order")
        self.add_option("hessian_approximation", "limited-memory")

        remaining = timer.remaining()

        if remaining <= 0.0:
            raise StepSolverError("Time limit reached")
        elif np.isfinite(remaining):
            self.add_option("max_cpu_time", remaining)

    def solve(self, timer):
        iterate = self.iterate

        self.set_options(timer)
        x0 = iterate.x

        # Solve using Ipopt
        x, info = super().solve(x0)

        if info["status"] != 0:
            raise StepSolverError("Ipopt failed to solve the problem")

        return x


class BoxReducedController(StepController):

    def objective(self, iterate, x, lamb, rho):
        xhat = iterate.x
        yhat = iterate.y

        obj = self.problem.obj(x)
        cons = self.problem.cons(x)

        dx = x - xhat
        dx_norm_sq = np.dot(dx, dx)

        w = -1 / lamb * cons
        dy = w - yhat
        dy_norm_sq = np.dot(dy, dy)

        curr_obj = obj + 0.5 * rho * np.dot(cons, cons)
        curr_obj += 0.5 * lamb * (dx_norm_sq + dy_norm_sq)

        return curr_obj

    def gradient(self, iterate, x, lamb, rho):
        xhat = iterate.x
        yhat = iterate.y

        obj_grad = self.problem.obj_grad(x)
        cons = self.problem.cons(x)
        cons_jac = self.problem.cons_jac(x)

        dx = x - xhat
        cons_jac_factor = (rho + (1 / lamb)) * cons + yhat
        return obj_grad + lamb * dx + cons_jac.T.dot(cons_jac_factor)

    def solve_step(self, iterate, rho, dt, timer):
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

    def step(
        self, iterate, rho: float, dt: float, next_steps, display: bool, timer
    ) -> StepControlResult:

        lamb = 1.0 / dt

        problem = self.problem
        params = self.params

        # reduced_problem = BoxReducedProblem(self, iterate, lamb, rho)
        # x = reduced_problem.solve(timer=timer)

        # TODO: The minimize function shipped with scipy
        # do not consistently produce high-quality solutions,
        # causing the optimization of the overall problem to fail.
        x = self.solve_step(iterate, rho, dt, timer=timer)

        cons = problem.cons(x)
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
