import abc
import numpy as np
import scipy as sp
from pygradflow.iterate import Iterate
from pygradflow.problem import Problem


class _Func(abc.ABC):
    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        self.problem = problem
        self.orig_iterate = iterate
        self.dt = dt

        self.n = problem.num_vars
        self.m = problem.num_cons

    def compute_active_set(self, x, lb, ub):
        """
        Compute the active set at the given point with respect to the
        given bounds. The active set has boolean entries, where a
        value of `True` at index `j`
        indicates that the input x_j is outside [lb_j, ub_j]
        and should be clipped to the interval in the projection.
        """
        assert (lb <= ub).all()

        return np.logical_or(x < lb - 1e-8, x > ub + 1e-8)

    def project(self, x, lb, ub, active_set):
        assert active_set.dtype == bool
        assert active_set.shape == lb.shape
        assert (lb <= ub).all()

        p = np.copy(x)

        p[active_set] = np.clip(x[active_set], lb[active_set], ub[active_set])

        assert (lb[active_set] <= p[active_set]).all()
        assert (p[active_set] <= ub[active_set]).all()

        return p

    @abc.abstractmethod
    def value_at(self, iterate, rho, active_set=None):
        raise NotImplementedError


class ImplicitFunc(_Func):
    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        super().__init__(problem, iterate, dt)

    def project(self, x, active_set):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub

        return super().project(x, lb, ub, active_set)

    def compute_active_set(self, x):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub
        return super().compute_active_set(x, lb, ub)

    def projection_initial(self, iterate, rho):
        x_0 = self.orig_iterate.x
        dt = self.dt
        return x_0 - dt * iterate.aug_lag_deriv_x(rho)

    def apply_project_deriv(self, mat, active_set):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub
        (num_rows, _) = mat.shape

        assert active_set.dtype == bool
        assert active_set.shape == lb.shape

        assert (lb <= ub).all()

        assert (lb != np.inf).all()
        assert (ub != -np.inf).all()

        assert (num_rows,) == lb.shape

        mat = mat.tocoo()

        inactive_set = np.logical_not(active_set)
        inactive_indices = np.where(inactive_set)[0]

        alive_indices = np.isin(mat.row, inactive_indices)

        assert inactive_set[mat.row[alive_indices]].all()

        next_rows = mat.row[alive_indices]
        next_cols = mat.col[alive_indices]
        next_entries = mat.data[alive_indices]

        proj_mat = sp.sparse.coo_matrix(
            (next_entries, (next_rows, next_cols)), mat.shape
        )

        assert inactive_set[proj_mat.row].all()

        return proj_mat

    def value_at(self, iterate, rho, active_set=None):
        y_0 = self.orig_iterate.y
        dt = self.dt

        p = self.projection_initial(iterate, rho)

        if active_set is None:
            active_set = self.compute_active_set(p)

        xval = iterate.x - self.project(p, active_set)
        yval = iterate.y - (y_0 + dt * iterate.aug_lag_deriv_y())
        return np.concatenate([xval, yval])

    def deriv(self, jac, hess, active_set):
        n = self.n
        m = self.m
        dt = self.dt

        assert active_set is not None
        params = self.orig_iterate.params

        F_11 = sp.sparse.eye(n, dtype=params.dtype)
        F_11 += self.apply_project_deriv(dt * hess, active_set)

        F_12 = self.apply_project_deriv(dt * jac.T, active_set)

        assert F_11.dtype == params.dtype
        assert F_12.dtype == params.dtype

        F_21 = -dt * jac
        F_22 = sp.sparse.eye(m, dtype=params.dtype)

        deriv = sp.sparse.bmat([[F_11, F_12], [F_21, F_22]],
                               format="csc")

        assert deriv.dtype == params.dtype

        return deriv

    def deriv_at(self, iterate, rho, active_set=None):
        if active_set is None:
            p = self.projection_initial(iterate, rho)
            active_set = self.compute_active_set(p)

        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        return self.deriv(jac, hess, active_set)


class ScaledImplicitFunc(_Func):
    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        super().__init__(problem, iterate, dt)
        self.lamb = 1.0 / dt
        params = iterate.params
        self.lb = (self.lamb * problem.var_lb).astype(params.dtype)
        self.ub = (self.lamb * problem.var_ub).astype(params.dtype)

    def value_at(self, iterate, rho, active_set=None):
        y_0 = self.orig_iterate.y
        lamb = self.lamb

        p = self.projection_initial(iterate, rho)

        if active_set is None:
            active_set = self.compute_active_set(p)

        xval = lamb * iterate.x - self.project(p, active_set)
        yval = -(lamb * iterate.y - (lamb * y_0 + iterate.aug_lag_deriv_y()))

        return np.concatenate([xval, yval])

    def projection_initial(self, iterate, rho):
        x_0 = self.orig_iterate.x
        lamb = self.lamb

        return lamb * x_0 - iterate.aug_lag_deriv_x(rho)

    def project(self, x, active_set):
        return super().project(x, self.lb, self.ub, active_set)

    def compute_active_set(self, x):
        return super().compute_active_set(x, self.lb, self.ub)
