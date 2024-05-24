import numpy as np

from pygradflow.iterate import Iterate

EPS = np.finfo(float).eps
xtol = 4 * EPS
atol = 4 * EPS


def lazy_func(func):
    values = [None]

    def wrapped(*args, **kwds):
        if values[0] is None:
            values[0] = func(*args, **kwds)
        return values[0]

    return wrapped


def is_pos(x):
    return x > 0.0 and not Flow.isclose(x, 0.0)


def is_neg(x):
    return x < 0.0 and not Flow.isclose(x, 0.0)


def func_pos(func, deriv, j):
    if is_pos(func()[j]):
        return True
    elif Flow.isclose(func()[j], 0.0):
        if is_pos(deriv()[j]):
            return True
    return False


def func_neg(func, deriv, j):
    if is_neg(func()[j]):
        return True
    elif Flow.isclose(func()[j], 0.0):
        if is_neg(deriv()[j]):
            return True
    return False


class Flow:
    def __init__(self, problem, params, eval):
        self.problem = problem
        self.params = params
        self.eval = eval

    @staticmethod
    def isclose(x, y):
        return np.isclose(x, y, rtol=xtol, atol=atol)

    @property
    def num_vars(self):
        return self.problem.num_vars

    @property
    def num_cons(self):
        return self.problem.num_cons

    def is_boxed(self, x):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub
        return (lb <= x).all() and (x <= ub).all()

    def is_approx_boxed(self, x):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub

        is_boxed_lb = np.logical_or(lb <= x, Flow.isclose(lb, x))
        is_boxed_ub = np.logical_or(x <= ub, Flow.isclose(ub, x))

        return np.logical_and(is_boxed_lb, is_boxed_ub).all()

    def rhs(self, z, rho, c=None):
        eval = self.eval
        (x, y) = self.split_states(z)

        if c is None:
            c = eval.cons(x)

        dx = self.neg_aug_lag_deriv_x(z, rho, c)
        dy = c
        return np.concatenate((dx, dy))

    def rhs_func(self, rho):
        def rhs(t, z):
            return self.rhs(z, rho)

        return rhs

    def aug_lag_deriv_x(self, z, rho, c=None):
        eval = self.eval
        (x, y) = self.split_states(z)

        if c is None:
            c = eval.cons(x)

        lhs = rho * c + y
        aug_lag_deriv_x = eval.obj_grad(x) + eval.cons_jac(x).T.dot(lhs)
        return aug_lag_deriv_x

    def aug_lag_deriv_y(self, z, rho, c=None):
        eval = self.eval
        (x, y) = self.split_states(z)

        if c is None:
            c = eval.cons(x)

        return c

    def neg_aug_lag_deriv_x(self, z, rho, c=None):
        return -self.aug_lag_deriv_x(z, rho, c)

    def obj(self, z):
        problem = self.problem
        (x, y) = self.split_states(z)
        return problem.obj(x)

    def neg_aug_lag_deriv_xx(self, z, rho, c=None):
        eval = self.eval
        problem = self.problem
        params = self.params

        (x, y) = self.split_states(z)

        iterate = Iterate(problem, params, x, y)
        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        if c is None:
            c = eval.cons(x)

        aug_lag_deriv_x = self.aug_lag_deriv_x(z, rho, c=c)
        aug_lag_deriv_y = c

        dt = hess @ aug_lag_deriv_x - jac.T @ aug_lag_deriv_y
        return dt

    def rhs_deriv_x(self, z, rho):
        return self.neg_aug_lag_deriv_xx(z, rho)

    def split_states(self, z):
        problem = self.problem
        num_vars = problem.num_vars
        num_cons = problem.num_cons
        assert z.shape == (num_vars + num_cons,)

        x = z[:num_vars]
        y = z[num_vars:]
        return x, y
