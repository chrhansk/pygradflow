import numpy as np
import scipy as sp

from pygradflow.integration.flow import Flow
from pygradflow.integration.problem_switches import ProblemSwitches
from pygradflow.iterate import Iterate
from pygradflow.util import keep_rows


class RestrictedFlow:
    def __init__(self, flow, filter):
        self.flow = flow
        self.problem = flow.problem
        self.params = flow.params
        self.eval = flow.eval
        problem = flow.problem
        num_vars = problem.num_vars

        assert filter.shape == (num_vars,)
        assert filter.dtype == bool

        self.filter = filter
        self.switches = ProblemSwitches(self)

    def check_point(self, z, rho):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub

        (x, y) = self.flow.split_states(z)
        at_lb = Flow.isclose(x, lb)
        at_ub = Flow.isclose(x, ub)

        rhs = self.rhs(z, rho)
        (rhs_x, _) = self.flow.split_states(rhs)

        num_vars = problem.num_vars
        filter = self.filter

        for j in range(num_vars):
            if at_lb[j]:
                if rhs_x[j] < 0.0 and not (Flow.isclose(rhs_x[j], 0.0)):
                    assert not filter[j]
                elif rhs_x[j] > 0.0 and not (Flow.isclose(rhs_x[j], 0.0)):
                    assert filter[j]
            elif at_ub[j]:
                if rhs_x[j] > 0.0 and not (Flow.isclose(rhs_x[j], 0.0)):
                    assert not filter[j]
                elif rhs_x[j] < 0.0 and not (Flow.isclose(rhs_x[j], 0.0)):
                    assert filter[j]
            else:
                assert filter[j]

    def create_event_triggers(self, z, rho):
        return self.switches.create_event_triggers(self.filter, z, rho)

    def rhs_deriv_x(self, z, rho, c=None):
        problem = self.problem
        params = self.params
        eval = self.eval

        (x, y) = self.flow.split_states(z)

        iterate = Iterate(problem, params, x, y)
        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        if c is None:
            c = eval.cons(x)

        aug_lag_deriv_x = self.flow.aug_lag_deriv_x(z, rho, c=c)
        # aug_lag_deriv_x *= self.filter
        aug_lag_deriv_y = c

        dt = hess @ aug_lag_deriv_x - jac.T @ aug_lag_deriv_y
        return dt

    def rhs(self, z, rho, c=None):
        (x, y) = self.flow.split_states(z)
        eval = self.eval

        if c is None:
            c = eval.cons(x)

        dx = self.flow.neg_aug_lag_deriv_x(z, rho, c)
        dx *= self.filter
        dy = c
        return np.concatenate((dx, dy))

    def rhs_func(self, rho):
        assert rho > 0

        def rhs(_, z):
            return self.rhs(z, rho)

        return rhs

    def rhs_jac(self, z, rho):
        problem = self.problem
        params = self.params

        (x, y) = self.flow.split_states(z)

        iterate = Iterate(problem, params, x, y)
        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        filtered_hess = keep_rows(hess, self.filter)
        filtered_jac = keep_rows(jac.T, self.filter)

        return sp.sparse.bmat(
            [[-filtered_hess, -filtered_jac], [jac, None]], format="csr"
        )

    def rhs_jac_func(self, rho):
        assert rho > 0

        def rhs_jac(_, z):
            return self.rhs_jac(z, rho)

        return rhs_jac

    def residuum(self, z):
        return np.linalg.norm(self.rhs(z, rho=0.0))
