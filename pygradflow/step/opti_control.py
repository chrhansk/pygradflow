import math

import cyipopt
import numpy as np

from pygradflow.eval import EvalError, astype
from pygradflow.iterate import Iterate
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    StepSolverError,
)


def map_cython_exception(func):
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ArithmeticError, EvalError) as e:
            raise cyipopt.CyIpoptEvaluationError() from e

    return wrapped_func


class ImplicitProblem(cyipopt.Problem):
    """
    Solves the primal-dual proximally regularized problem
     .. math::
        \\begin{align}
            \\min_{x \\in \\mathbb{R}^{n}, w \\in \\mathbb{R}^{n}} \\quad
            & f(x) + \\frac{\\rho}{2} \\|c(x)\\|^{2} + \\frac{\\lambda}{2} (\\| x - \\hat{x} \\|^{2} + \\| w - \\hat{y} \\|^{2}) \\\\
            \\text{s.t.} \\quad & 0 = c(x) + \\lambda w \\\\
                              & l^x \\leq x \\leq u^x
        \\end{align}

    whose optimum coincides with the implicit Euler step of the augmented Lagrangian
    of the original problem. The problem is solved using Ipopt with a limited-memory
    BFGS Hessian approximation.
    """

    def __init__(self, iterate, lamb, rho, rescaled=True):
        # Resacled:
        # Replace w by v = w / sqrt(lambda) in the augmented Lagrangian
        # to avoid numerical issues
        assert lamb >= 0
        assert rho >= 0

        self.iterate = iterate
        self.eval = iterate.eval
        self.lamb = lamb
        self.rho = rho

        self.rescaled = rescaled

        problem = iterate.problem
        prob_num_vars = problem.num_vars
        prob_num_cons = problem.num_cons

        num_vars = prob_num_vars + prob_num_cons
        num_cons = prob_num_cons

        self.var_lb = np.concatenate([problem.var_lb, np.full(prob_num_cons, -np.inf)])
        self.var_ub = np.concatenate([problem.var_ub, np.full(prob_num_cons, np.inf)])

        self.cons_lb = np.zeros(prob_num_cons)
        self.cons_ub = np.zeros(prob_num_cons)

        self.problem = problem
        self.prob_num_vars = prob_num_vars
        self.prob_num_cons = prob_num_cons

        self.num_vars = num_vars
        self.num_cons = num_cons

        super(ImplicitProblem, self).__init__(
            n=num_vars,
            m=num_cons,
            lb=self.var_lb,
            ub=self.var_ub,
            cl=self.cons_lb,
            cu=self.cons_ub,
        )

    @map_cython_exception
    def objective(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        rho = self.rho
        lamb = self.lamb

        eval = self.eval
        obj = eval.obj(x)
        cons = eval.cons(x)
        aug_obj = obj + (rho / 2) * np.dot(cons, cons)

        xdiff = x - self.iterate.x

        if self.rescaled:
            wdiff = w - math.sqrt(lamb) * self.iterate.y
            aug_obj += (lamb / 2) * np.dot(xdiff, xdiff) + (1 / 2) * np.dot(
                wdiff, wdiff
            )
        else:
            wdiff = w - self.iterate.y
            aug_obj += (lamb / 2) * (np.dot(xdiff, xdiff) + np.dot(wdiff, wdiff))

        return float(aug_obj)

    @map_cython_exception
    def gradient(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        eval = self.eval
        obj_grad = eval.obj_grad(x)
        cons_jac = eval.cons_jac(x)
        cons = eval.cons(x)

        rho = self.rho
        lamb = self.lamb

        cons_prod = cons_jac.T.dot(cons)

        gradx = obj_grad + rho * cons_prod + lamb * (x - self.iterate.x)

        if self.rescaled:
            gradw = w - math.sqrt(lamb) * self.iterate.y
        else:
            gradw = lamb * (w - self.iterate.y)
        grad = np.concatenate([gradx, gradw])

        assert grad.shape == (self.num_vars,)

        return astype(grad, np.float64)

    @map_cython_exception
    def constraints(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        eval = self.eval
        prob_cons = eval.cons(x)
        lamb = self.lamb

        if self.rescaled:
            cons = prob_cons + math.sqrt(lamb) * w
        else:
            cons = prob_cons + lamb * w

        assert cons.shape == (self.num_cons,)

        return astype(cons, np.float64)

    @map_cython_exception
    def jacobian(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        eval = self.eval
        problem = self.problem
        jac_x = eval.cons_jac(x).tocoo()
        data_x = jac_x.data

        if self.rescaled:
            data_w = np.full((problem.num_cons,), math.sqrt(self.lamb))
        else:
            data_w = np.full((problem.num_cons,), self.lamb)

        assert self._jac_nnz is not None

        data = np.concatenate([data_x, data_w])

        assert data.shape == (self._jac_nnz,)

        return astype(data, np.float64)

    def jacobianstructure(self):
        problem = self.problem
        iterate = self.iterate
        x = iterate.x

        eval = self.eval
        jac_x = eval.cons_jac(x).tocoo()

        (r_x, c_x) = (jac_x.row, jac_x.col)
        (r_w, c_w) = np.diag_indices(problem.num_cons)
        r_w = np.copy(r_w)
        c_w = np.copy(c_w)

        c_w += problem.num_vars

        rows = np.concatenate([r_x, r_w])
        cols = np.concatenate([c_x, c_w])

        self._jac_nnz = rows.size

        assert rows.shape == cols.shape
        assert rows.ndim == 1

        assert (rows >= 0).all()
        assert (cols >= 0).all()
        assert (rows < self.num_cons).all()
        assert (cols < self.num_vars).all()

        assert rows.dtype == np.int64
        assert cols.dtype == np.int64

        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        raise NotImplementedError()

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
        y0 = iterate.y
        z0 = np.concatenate([x0, y0])

        # Solve using Ipopt
        try:
            z, info = super().solve(z0)
            assert np.isfinite(z).all()
        except EvalError:
            raise StepSolverError("Failed to evaluate subproblem")

        if info["status"] not in [0, 1]:
            raise StepSolverError("Ipopt failed to solve subproblem")

        x = z[: self.prob_num_vars]
        y = info["mult_g"]

        return (x, y)


class OptimizingController(StepController):
    def step(self, iterate, rho, dt, display, timer) -> StepControlResult:

        problem = self.problem
        params = self.params

        lamb = 1.0 / dt
        implicit_problem = ImplicitProblem(iterate, lamb, rho)
        (x, y) = implicit_problem.solve(timer=timer)

        next_iterate = Iterate(problem, params, x, y, iterate.eval)

        return StepControlResult(
            next_iterate, 0.5 * lamb, active_set=None, rcond=None, accepted=True
        )
