import numpy as np
import scipy as sp

from pygradflow.problem import Problem
from pygradflow.util import sparse_zero


class ConstrainedProblem(Problem):
    def __init__(self, problem):
        cons_lb = problem.cons_lb
        cons_ub = problem.cons_ub
        self.problem = problem
        self.create_slacks(cons_lb, cons_ub)

        var_lb = problem.var_lb
        var_ub = problem.var_ub

        num_slacks = len(self.slack_positions)

        slack_lb = cons_lb[self.slack_positions]
        slack_ub = cons_ub[self.slack_positions]

        if num_slacks > 0:
            var_lb = np.concatenate([var_lb, slack_lb])

            var_ub = np.concatenate([var_ub, slack_ub])

        num_cons = problem.num_cons
        super().__init__(var_lb, var_ub, num_cons=num_cons)

    def create_slacks(self, cons_lb, cons_ub):
        (num_cons,) = cons_lb.shape
        assert cons_lb.shape == cons_ub.shape

        has_offsets = False
        cons_offsets = np.zeros((num_cons,))

        slack_positions = []

        for i, (lb, ub) in enumerate(zip(cons_lb, cons_ub)):
            if lb == ub:
                if lb != 0.0:
                    has_offsets = True
                    cons_offsets[i] = -lb
                continue

            slack_positions.append(i)

        self.slack_positions = np.array(slack_positions, dtype=int)

        self.cons_offsets = None

        if has_offsets:
            self.cons_offsets = cons_offsets

    def orig_vals(self, x):
        orig_num_vars = self.problem.num_vars
        return x[:orig_num_vars]

    def slack_vals(self, x):
        orig_num_vars = self.problem.num_vars
        return x[orig_num_vars:]

    def obj(self, x):
        return self.problem.obj(self.orig_vals(x))

    def obj_grad(self, x):
        orig_grad = self.problem.obj_grad(self.orig_vals(x))

        num_slacks = len(self.slack_positions)

        if num_slacks == 0:
            return orig_grad

        return np.concatenate([orig_grad, np.zeros((num_slacks,))])

    def cons(self, x):
        orig_cons = self.problem.cons(self.orig_vals(x))

        num_slacks = len(self.slack_positions)

        if self.cons_offsets is not None:
            orig_cons += self.cons_offsets

        if num_slacks == 0:
            return orig_cons

        slack_vals = self.slack_vals(x)

        for pos, val in zip(self.slack_positions, slack_vals):
            orig_cons[pos] -= val

        return orig_cons

    def cons_jac(self, x):
        orig_cons_jac = self.problem.cons_jac(self.orig_vals(x))

        num_slacks = len(self.slack_positions)

        if num_slacks == 0:
            return orig_cons_jac

        data = np.full((num_slacks,), fill_value=-1.0, dtype=float)
        cols = np.arange(num_slacks)
        rows = self.slack_positions
        num_cons = self.num_cons

        additional_cons_jac = sp.sparse.coo_matrix(
            (data, (rows, cols)), shape=(num_cons, num_slacks)
        )

        return sp.sparse.bmat([[orig_cons_jac, additional_cons_jac]])

    def lag_hess(self, x, y):
        orig_hess = self.problem.lag_hess(self.orig_vals(x), y)

        (num_vars, _) = orig_hess.shape

        num_slacks = len(self.slack_positions)

        if num_slacks == 0:
            return orig_hess

        lower = sparse_zero(shape=(num_slacks, num_vars))
        diag = sparse_zero(shape=(num_slacks, num_slacks))

        return sp.sparse.bmat([[orig_hess, lower.T], [lower, diag]])

    def transform_sol(self, orig_x, orig_y):
        assert orig_x.shape == (self.problem.num_vars,)
        assert orig_y.shape == (self.problem.num_cons,)

        num_slacks = len(self.slack_positions)

        if num_slacks == 0:
            return (orig_x, orig_y)

        slack_vals = np.zeros((num_slacks,))

        orig_cons_vals = self.problem.cons(orig_x)

        for i, pos in enumerate(self.slack_positions):
            slack_vals[i] = orig_cons_vals[pos]

        x = np.concatenate([orig_x, slack_vals])
        y = orig_y

        return (x, y)

    def restore_sol(self, x, y, d):
        assert x.shape == (self.num_vars,)
        assert y.shape == (self.num_cons,)
        assert d.shape == (self.num_vars,)

        num_slacks = len(self.slack_positions)

        if num_slacks == 0:
            return (x, y, d)

        orig_x = self.orig_vals(x)
        orig_y = y
        orig_d = self.orig_vals(d)

        return (orig_x, orig_y, orig_d)
