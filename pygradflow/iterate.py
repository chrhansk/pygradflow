import numpy as np

from pygradflow.util import norm_mult
from pygradflow.lazy import lazyprop
from pygradflow.active_set import ActiveSet


def _read_only(a):
    a.flags.writeable = False
    return a


class Iterate:
    def __init__(self, problem, params, x, y):
        assert x.shape == (problem.num_vars,)
        assert y.shape == (problem.num_cons,)
        self.x = _read_only(np.copy(x))
        self.y = _read_only(np.copy(y))
        self.params = params
        self.problem = problem

    def copy(self):
        return Iterate(self.problem, self.params, np.copy(self.x), np.copy(self.y))

    @lazyprop
    def obj(self):
        return self.problem.obj(self.x)

    @lazyprop
    def obj_grad(self):
        return _read_only(self.problem.obj_grad(self.x))

    @lazyprop
    def cons(self):
        return _read_only(self.problem.cons(self.x))

    @lazyprop
    def cons_jac(self):
        return self.problem.cons_jac(self.x)

    def lag_hess(self, lag):
        return self.problem.lag_hess(self.x, lag)

    def aug_lag_violation(self, rho):
        cv = self.cons
        return rho / 2.0 * np.dot(cv, cv)

    def aug_lag_dual(self):
        cv = self.cons
        return np.dot(cv, self.y)

    def aug_lag(self, rho):
        o = self.obj
        cv = self.cons
        return o + rho / 2.0 * np.dot(cv, cv) + np.dot(cv, self.y)

    def aug_lag_deriv_x(self, rho):
        lhs = rho * self.cons + self.y

        return self.obj_grad + self.cons_jac.T.dot(lhs)

    def aug_lag_deriv_y(self):
        return self.cons

    def aug_lag_deriv_xy(self):
        return self.cons_jac

    def aug_lag_deriv_xx(self, rho):
        jac = self.cons_jac
        cons = self.cons
        m = self.y + rho * cons

        if rho == 0.0:
            return self.lag_hess(m)
        else:
            return self.lag_hess(m) + rho * np.dot(jac.T, jac)

    def dist(self, other):
        return norm_mult(self.x - other.x, self.y - other.y)

    @lazyprop
    def active_set(self):
        return ActiveSet(self)

    @lazyprop
    def bound_duals(self):
        r = -(self.obj_grad + self.cons_jac.T.dot(self.y))
        d = np.zeros_like(self.x)

        active_set = self.active_set

        d[active_set.at_upper] = np.maximum(r[active_set.at_upper], 0.0)
        d[active_set.at_lower] = np.minimum(r[active_set.at_lower], 0.0)
        d[active_set.at_both] = r[active_set.at_both]

        return d

    @lazyprop
    def bound_violation(self):
        lb = self.problem.var_lb
        ub = self.problem.var_ub
        x = self.x

        lower = np.linalg.norm(np.maximum(lb - x, 0.0), np.inf)
        upper = np.linalg.norm(np.maximum(x - ub, 0.0), np.inf)

        return max(lower, upper)

    @lazyprop
    def cons_violation(self):
        c = self.cons
        if c.size == 0:
            return 0.0
        return np.linalg.norm(c, np.inf)

    @lazyprop
    def stat_res(self):
        r = self.obj_grad + self.cons_jac.T.dot(self.y) + self.bound_duals
        return np.linalg.norm(r, np.inf)

    @property
    def total_res(self):
        return max(self.cons_violation, self.bound_violation, self.stat_res)
