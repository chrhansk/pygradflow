import abc
from typing import Optional

import numpy as np
import scipy as sp

from pygradflow.iterate import Iterate
from pygradflow.problem import Problem
from pygradflow.util import keep_rows


class StepFunc(abc.ABC):
    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        self.problem = problem
        self.orig_iterate = iterate
        self.dt = dt

        self.n = problem.num_vars
        self.m = problem.num_cons

    def compute_active_set_box(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray):
        """
        Compute the active set at the given point with respect to the
        given bounds.

        Parameters
        ----------

        x: np.ndarray
            A primal point in :math:`\\mathbb{R}^{n}`
        lb, ub: np.ndarray
            Bounds in :math:`\\mathbb{R}^{n}`

        Returns
        -------
        np.ndarray
            A vector with boolean entries, where a value of `True`
            at index `j` indicates that :math:`x_j` is
            outside the interval  :math:`[l_j, u_j]` and should be
            clipped to it during the projection.
        """
        assert (lb <= ub).all()

        return np.logical_or(x < lb - 1e-8, x > ub + 1e-8)

    def project_box(
        self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray, active_set: np.ndarray
    ) -> np.ndarray:
        assert active_set.dtype == bool
        assert active_set.shape == lb.shape
        assert (lb <= ub).all()

        p = np.copy(x)

        p[active_set] = np.clip(x[active_set], lb[active_set], ub[active_set])

        assert (lb[active_set] <= p[active_set]).all()
        assert (p[active_set] <= ub[active_set]).all()

        return p

    @abc.abstractmethod
    def value_at(
        self, iterate: Iterate, rho, active_set: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def projection_initial(self, iterate: Iterate, rho: float, tau=None) -> np.ndarray:
        raise NotImplementedError()

    def compute_active_set(self, iterate: Iterate, rho: float, tau=None) -> np.ndarray:
        p = self.projection_initial(iterate, rho, tau)
        return self.active_set_at_point(p)

    @abc.abstractmethod
    def active_set_at_point(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def apply_project_deriv(self, mat: sp.sparse.spmatrix, active_set: np.ndarray):
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

        inactive_set = np.logical_not(active_set)
        proj_mat = keep_rows(mat, inactive_set)

        return proj_mat


class ImplicitFunc(StepFunc):
    """
    Standard residual function for implicit Euler steps
    with the value

    .. math::
        F(x, y; \\hat{x}, \\hat{y}) =
        \\begin{pmatrix}
            x - P_{C} (\\hat{x} - \\Delta t \\nabla_{x} \\mathcal{L}^{\\rho}(x, y)) \\\\
            y - (\\hat{y} + \\Delta t \\nabla_{y} \\mathcal{L}^{\\rho}(x, y)
        \\end{pmatrix}

    The values :math:`\\hat{x}` and :math:`\\hat{y}` are obtained
    from the iteratre provided in the constructor
    """

    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        super().__init__(problem, iterate, dt)

    def project(self, x: np.ndarray, active_set: np.ndarray):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub

        return super().project_box(x, lb, ub, active_set)

    def active_set_at_point(self, p):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub
        return super().compute_active_set_box(p, lb, ub)

    def projection_initial(self, iterate: Iterate, rho: float, tau=None):
        x_0 = self.orig_iterate.x
        dt = self.dt

        if tau is not None:
            lamb = 1.0 / dt
            x = iterate.x
            return (
                (1.0 - tau * lamb) * x
                + (tau * lamb) * x_0
                - tau * iterate.aug_lag_deriv_x(rho)
            )

        return x_0 - dt * iterate.aug_lag_deriv_x(rho)

    # @override
    def value_at(self, iterate, rho, active_set=None):
        y_0 = self.orig_iterate.y
        dt = self.dt

        p = self.projection_initial(iterate, rho)

        if active_set is None:
            active_set = self.compute_active_set(iterate, rho)

        xval = iterate.x - self.project(p, active_set)
        yval = iterate.y - (y_0 + dt * iterate.aug_lag_deriv_y())
        return np.concatenate([xval, yval])

    def deriv(
        self, jac: sp.sparse.spmatrix, hess: sp.sparse.spmatrix, active_set: np.ndarray
    ):
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

        deriv = sp.sparse.bmat([[F_11, F_12], [F_21, F_22]], format="csc")

        assert deriv.dtype == params.dtype

        return deriv

    def deriv_at(
        self, iterate: Iterate, rho: float, active_set: Optional[np.ndarray] = None
    ):
        if active_set is None:
            active_set = self.compute_active_set(iterate, rho)

        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        return self.deriv(jac, hess, active_set)


class ScaledImplicitFunc(StepFunc):
    """
    The *scaled* residual function for implicit Euler steps
    is defined as the normal residual function
    given by :py:class:`pygradflow.implicit_func.ImplicitFunc`
    scaled by a factor
    of :math:`\\lambda = 1 / \\Delta t`.
    """

    def __init__(self, problem: Problem, iterate: Iterate, dt: float) -> None:
        super().__init__(problem, iterate, dt)
        self.lamb = 1.0 / dt
        params = iterate.params
        self.lb = (self.lamb * problem.var_lb).astype(params.dtype)
        self.ub = (self.lamb * problem.var_ub).astype(params.dtype)

    # @override
    def value_at(self, iterate, rho, active_set=None):
        y_0 = self.orig_iterate.y
        lamb = self.lamb

        p = self.projection_initial(iterate, rho)

        if active_set is None:
            active_set = self.compute_active_set(iterate, rho)

        xval = lamb * iterate.x - self.project(p, active_set)
        yval = -(lamb * iterate.y - (lamb * y_0 + iterate.aug_lag_deriv_y()))

        return np.concatenate([xval, yval])

    def projection_initial(self, iterate: Iterate, rho: float, tau=None):
        x_0 = self.orig_iterate.x
        lamb = self.lamb

        if tau is not None:
            lamb = 1.0 / self.dt
            x = iterate.x
            f_x = lamb * (1 - tau * lamb)
            f_x0 = tau * lamb * lamb
            f_d = tau * lamb

            return f_x * x + f_x0 * x_0 - f_d * iterate.aug_lag_deriv_x(rho)

        return lamb * x_0 - iterate.aug_lag_deriv_x(rho)

    def project(self, x: np.ndarray, active_set: np.ndarray):
        return super().project_box(x, self.lb, self.ub, active_set)

    def active_set_at_point(self, p):
        return super().compute_active_set_box(p, self.lb, self.ub)

    def deriv(
        self, jac: sp.sparse.spmatrix, hess: sp.sparse.spmatrix, active_set: np.ndarray
    ):
        n = self.n
        m = self.m
        dt = self.dt
        lamb = 1.0 / dt

        assert active_set is not None
        params = self.orig_iterate.params

        F_11 = sp.sparse.diags([lamb], shape=(n, n), dtype=params.dtype)

        F_11 = lamb * sp.sparse.eye(n, dtype=params.dtype)
        F_11 += self.apply_project_deriv(hess, active_set)

        F_12 = self.apply_project_deriv(jac.T, active_set)

        assert F_11.dtype == params.dtype
        assert F_12.dtype == params.dtype

        F_21 = -jac

        F_22 = sp.sparse.diags([lamb], shape=(m, m), dtype=params.dtype)

        deriv = sp.sparse.bmat([[F_11, F_12], [F_21, F_22]], format="csc")

        assert deriv.dtype == params.dtype

        return deriv

    def deriv_at(
        self, iterate: Iterate, rho: float, active_set: Optional[np.ndarray] = None
    ):
        if active_set is None:
            active_set = self.compute_active_set(iterate, rho)

        hess = iterate.aug_lag_deriv_xx(rho)
        jac = iterate.aug_lag_deriv_xy()

        return self.deriv(jac, hess, active_set)
