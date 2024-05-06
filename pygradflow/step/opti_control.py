import math

import cyipopt
import numpy as np

from pygradflow.eval import astype
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.step.step_control import (
    StepController,
    StepControlResult,
    StepSolverError,
)

# Replace w by v = w / sqrt(lambda) in the augmented Lagrangian
# to avoid numerical issues
rescaled = True


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

    def __init__(self, iterate, lamb, rho):
        assert lamb >= 0
        assert rho >= 0

        self.iterate = iterate
        self.eval = iterate.eval
        self.lamb = lamb
        self.rho = rho

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

        if rescaled:
            wdiff = w - math.sqrt(lamb) * self.iterate.y
            aug_obj += (lamb / 2) * np.dot(xdiff, xdiff) + (1 / 2) * np.dot(
                wdiff, wdiff
            )
        else:
            wdiff = w - self.iterate.y
            aug_obj += (lamb / 2) * (np.dot(xdiff, xdiff) + np.dot(wdiff, wdiff))

        assert np.isfinite(aug_obj)

        return float(aug_obj)

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

        if rescaled:
            gradw = w - math.sqrt(lamb) * self.iterate.y
        else:
            gradw = lamb * (w - self.iterate.y)
        grad = np.concatenate([gradx, gradw])

        assert grad.shape == (self.num_vars,)
        assert np.isfinite(grad).all()

        return astype(grad, np.float64)

    def constraints(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        eval = self.eval
        prob_cons = eval.cons(x)
        lamb = self.lamb

        if rescaled:
            cons = prob_cons + math.sqrt(lamb) * w
        else:
            cons = prob_cons + lamb * w

        assert cons.shape == (self.num_cons,)
        assert np.isfinite(cons).all()

        return astype(cons, np.float64)

    def jacobian(self, z):
        assert z.shape == (self.num_vars,)
        (x, w) = np.split(z, [self.prob_num_vars])

        eval = self.eval
        problem = self.problem
        jac_x = eval.cons_jac(x).tocoo()
        data_x = jac_x.data

        if rescaled:
            data_w = np.full((problem.num_cons,), math.sqrt(self.lamb))
        else:
            data_w = np.full((problem.num_cons,), self.lamb)

        assert self._jac_nnz is not None

        data = np.concatenate([data_x, data_w])

        assert data.shape == (self._jac_nnz,)
        assert np.isfinite(data).all()

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

    def set_options(self):
        import logging

        logging.getLogger("cyipopt").setLevel(logging.WARNING)
        self.add_option("print_level", 0)
        self.add_option("derivative_test", "first-order")
        self.add_option("hessian_approximation", "limited-memory")

    def solve(self):
        iterate = self.iterate

        self.set_options()
        x0 = iterate.x
        y0 = iterate.y
        z0 = np.concatenate([x0, y0])

        # Solve using Ipopt
        z, info = super().solve(z0)

        assert np.isfinite(z).all()

        if info["status"] != 0:
            raise StepSolverError("Ipopt failed to solve the problem")

        x = z[: self.prob_num_vars]
        if rescaled:
            v = z[self.prob_num_vars :]
            w = v / math.sqrt(self.lamb)
        else:
            w = z[self.prob_num_vars :]
        y = iterate.y - w

        return (x, y)


class OptimizingController(StepController):
    def step(
        self,
        iterate,
        rho: float,
        dt: float,
        next_steps,
        display: bool,
    ) -> StepControlResult:

        problem = self.problem
        params = self.params

        lamb = 1.0 / dt
        implicit_problem = ImplicitProblem(iterate, lamb, rho)
        (x, y) = implicit_problem.solve()

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
