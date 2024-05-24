import functools
from typing import Optional

import numpy as np
import scipy as sp

from pygradflow.active_set import ActiveSet
from pygradflow.eval import Evaluator, SimpleEvaluator
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.util import norm_mult


def _read_only(a):
    a.flags.writeable = False
    return a


class Iterate:
    def __init__(
        self,
        problem: Problem,
        params: Params,
        x: np.ndarray,
        y: np.ndarray,
        eval: Optional[Evaluator] = None,
    ):
        assert x.shape == (problem.num_vars,)
        assert y.shape == (problem.num_cons,)
        self.x = _read_only(np.copy(x))
        self.y = _read_only(np.copy(y))
        self.params = params

        if eval:
            self.eval = eval
        else:
            self.eval = SimpleEvaluator(problem, params)
        self.problem = problem

    @property
    def z(self):
        return np.concatenate((self.x, self.y))

    def copy(self) -> "Iterate":
        return Iterate(
            self.problem, self.params, np.copy(self.x), np.copy(self.y), self.eval
        )

    def clipped(self) -> "Iterate":
        var_lb = self.problem.var_lb
        var_ub = self.problem.var_ub
        x = self.x
        if np.all(x >= var_lb) and np.all(x <= var_ub):
            return self
        xclip = np.empty_like(x)
        np.clip(x, var_lb, var_ub, out=xclip)
        return Iterate(self.problem, self.params, xclip, self.y, self.eval)

    @functools.cached_property
    def obj(self) -> float:
        return self.eval.obj(self.x)

    @functools.cached_property
    def obj_grad(self) -> np.ndarray:
        return _read_only(self.eval.obj_grad(self.x))

    @functools.cached_property
    def cons(self) -> np.ndarray:
        return _read_only(self.eval.cons(self.x))

    @functools.cached_property
    def cons_jac(self) -> sp.sparse.spmatrix:
        return self.eval.cons_jac(self.x)

    def lag_hess(self, y: np.ndarray) -> sp.sparse.spmatrix:
        return self.eval.lag_hess(self.x, y)

    def aug_lag_violation(self, rho: float) -> float:
        cv = self.cons
        return rho / 2.0 * np.dot(cv, cv)

    def aug_lag_dual(self) -> float:
        cv = self.cons
        return np.dot(cv, self.y)

    def aug_lag(self, rho: float) -> float:
        o = self.obj
        cv = self.cons
        return o + rho / 2.0 * np.dot(cv, cv) + np.dot(cv, self.y)

    def aug_lag_deriv_x(self, rho: float) -> np.ndarray:
        lhs = rho * self.cons + self.y

        return self.obj_grad + self.cons_jac.T.dot(lhs)

    def aug_lag_deriv_y(self) -> np.ndarray:
        return self.cons

    def aug_lag_deriv_xy(self) -> sp.sparse.spmatrix:
        return self.cons_jac

    def aug_lag_deriv_xx(self, rho: float) -> sp.sparse.spmatrix:
        jac = self.cons_jac
        cons = self.cons
        m = self.y + rho * cons

        if rho == 0.0:
            return self.lag_hess(m)
        else:
            return self.lag_hess(m) + rho * np.dot(jac.T, jac)

    def dist(self, other: "Iterate") -> float:
        return norm_mult(self.x - other.x, self.y - other.y)

    def locally_infeasible(self, feas_tol: float, local_infeas_tol: float) -> bool:
        """
        Check if the iterate is locally infeasible. It is
        judged to be locally infeasible if the constraint
        violation is greater than the tolerance and
        optimality conditions for the minimization
        of the constraint violation are (approximately) satisfied.
        """
        if self.cons_violation <= feas_tol:
            return False

        infeas_opt_res = self.cons_jac.T.dot(self.cons)

        at_lower = self.active_set.at_lower
        at_upper = self.active_set.at_upper

        infeas_opt_res[at_lower] = np.minimum(infeas_opt_res[at_lower], 0.0)
        infeas_opt_res[at_upper] = np.maximum(infeas_opt_res[at_upper], 0.0)

        return bool(np.linalg.norm(infeas_opt_res, ord=np.inf) <= local_infeas_tol)

    @functools.cached_property
    def active_set(self) -> ActiveSet:
        return ActiveSet(self)

    @functools.cached_property
    def bounds_dual(self) -> np.ndarray:
        r = -(self.obj_grad + self.cons_jac.T.dot(self.y))
        d = np.zeros_like(self.x)

        active_set = self.active_set

        d[active_set.at_upper] = np.maximum(r[active_set.at_upper], 0.0)
        d[active_set.at_lower] = np.minimum(r[active_set.at_lower], 0.0)
        d[active_set.at_both] = r[active_set.at_both]

        return d

    @functools.cached_property
    def bound_violation(self) -> float:
        lb = self.problem.var_lb
        ub = self.problem.var_ub
        x = self.x

        lower = float(np.linalg.norm(np.maximum(lb - x, 0.0), np.inf))
        upper = float(np.linalg.norm(np.maximum(x - ub, 0.0), np.inf))

        return max(lower, upper)

    @functools.cached_property
    def cons_violation(self) -> float:
        c = self.cons
        if c.size == 0:
            return 0.0
        return float(np.linalg.norm(c, np.inf))

    @functools.cached_property
    def stat_res(self) -> float:
        r = self.obj_grad + self.cons_jac.T.dot(self.y) + self.bounds_dual
        return float(np.linalg.norm(r, np.inf))

    def is_feasible(self, tol):
        return (self.cons_violation <= tol) and (self.bound_violation <= tol)

    @property
    def total_res(self) -> float:
        return max(self.cons_violation, self.bound_violation, self.stat_res)

    def obj_nonlin(self, other: "Iterate") -> float:
        dx = other.x - self.x
        next_obj = self.obj + np.dot(dx, self.obj_grad)
        dx_dot = np.dot(dx, dx)
        if np.isclose(dx_dot, 0.0):
            return 0.0
        return abs(other.obj - next_obj) / dx_dot

    def cons_nonlin(self, other: "Iterate") -> np.ndarray:
        dx = other.x - self.x
        next_cons = self.cons + self.cons_jac.dot(dx)
        dx_dot = np.dot(dx, dx)
        if np.isclose(dx_dot, 0.0):
            problem = self.problem
            return np.zeros((problem.num_cons,))
        return (other.cons - next_cons) / dx_dot
