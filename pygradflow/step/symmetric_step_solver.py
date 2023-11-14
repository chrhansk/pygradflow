import numpy as np
import scipy as sp

from pygradflow.step.scaled_step_solver import ScaledStepSolver
from pygradflow.iterate import Iterate
from pygradflow.params import Params
from pygradflow.problem import Problem


class SymmetricStepSolver(ScaledStepSolver):
    def __init__(
        self,
        problem: Problem,
        params: Params,
        orig_iterate: Iterate,
        dt: float,
        rho: float,
    ) -> None:
        super().__init__(problem, params, orig_iterate, dt, rho)

        assert dt > 0.0
        assert rho > 0.0

        self.active_set = None
        self.jac = None
        self.hess = None

    def compute_hess_jac(self) -> None:
        inactive_indices = np.where(np.logical_not(self.active_set))[0]

        lamb = 1.0 / self.dt

        n = self.n

        hess = self.hess + sp.sparse.diags([lamb],
                                           shape=(n, n),
                                           dtype=self.params.dtype)

        hess_rows = hess.tocsr()[inactive_indices, :]
        self.hess_rows = hess_rows.tocsc()

    def update_derivs(self, iterate: Iterate) -> None:
        super().update_derivs(iterate)
        self.jac = self.jac.tocsc()

    def reset_deriv(self) -> None:
        super().reset_deriv()
        self.hess_rows = None
        self.deriv = None

    def compute_deriv(self, active_set: np.ndarray) -> sp.sparse.spmatrix:
        inactive_indices = np.where(np.logical_not(self.active_set))[0]
        lamb = 1.0 / self.dt
        rho = self.rho

        m = self.m

        inactive_jac = self.jac[:, inactive_indices]
        inactive_hess = self.hess_rows[:, inactive_indices]

        lower_mat = sp.sparse.diags([-lamb / (1.0 + lamb * rho)],
                                    shape=(m, m),
                                    dtype=self.params.dtype)

        deriv = sp.sparse.bmat(
            [
                [inactive_hess, inactive_jac.T],
                [inactive_jac, lower_mat],
            ],
            format="csc",
        )

        assert deriv.dtype == self.params.dtype

        # May not be the case if Hessian itself is not
        # numerically symmetric
        # assert np.allclose((deriv - deriv.T).data, 0.0)

        return deriv

    def compute_rhs(
        self,
        active_indices: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
        b2t: np.ndarray,
    ) -> np.ndarray:
        active_hess = self.hess_rows[:, active_indices]
        active_jac = self.jac[:, active_indices]

        b1c = b1 - (active_hess @ b0)
        b2c = b2t - (active_jac @ b0)

        rhs = np.concatenate((b1c, b2c))

        return rhs

    def solve_scaled(self, b0, b1, b2t):
        params = self.params

        n = self.n
        m = self.m

        active_indices = np.where(self.active_set)[0]
        inactive_indices = np.where(np.logical_not(self.active_set))[0]

        inactive_set_size = inactive_indices.size

        self.compute_hess_jac()

        rhs = self.compute_rhs(active_indices, b0, b1, b2t)

        assert rhs.shape == (inactive_set_size + m,)

        s = self.solve_active_set(self.active_set, rhs)

        inactive_dx = s[:inactive_set_size]
        dy = s[inactive_set_size:]

        dx = np.zeros((n,), dtype=self.params.dtype)

        dx[inactive_indices] = inactive_dx
        dx[active_indices] = b0

        rcond = None
        if params.report_rcond:
            rcond = self.estimate_rcond(self.deriv, self.solver)

        return (dx, dy, rcond)

    def solve_deriv(
        self, active_set: np.ndarray, deriv: sp.sparse.spmatrix, rhs: np.ndarray
    ) -> np.ndarray:
        if self.solver is None:
            self.solver = self.linear_solver(self.deriv)

        return self.solver.solve(rhs)

    def solve_active_set(self, active_set: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if self.deriv is None:
            self.deriv = self.compute_deriv(self.active_set)

        return self.solve_deriv(self.active_set, self.deriv, rhs)
