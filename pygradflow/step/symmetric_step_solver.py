import numpy as np
import scipy as sp

from pygradflow.step.scaled_step_solver import ScaledStepSolver


class SymmetricStepSolver(ScaledStepSolver):
    def __init__(self, problem, params, orig_iterate, dt, rho) -> None:
        super().__init__(problem, params, orig_iterate, dt, rho)

        assert dt > 0.0
        assert rho > 0.0

        self.active_set = None
        self.jac = None
        self.hess = None
        self.dt = dt
        self.rho = rho

    def compute_hess_jac(self):
        inactive_indices = np.where(np.logical_not(self.active_set))[0]

        lamb = 1.0 / self.dt

        n = self.n

        hess = self.hess + sp.sparse.diags([lamb], shape=(n, n))

        hess_rows = hess.tocsr()[inactive_indices, :]
        self.hess_rows = hess_rows.tocsc()

    def update_derivs(self, iterate):
        super().update_derivs(iterate)
        self.jac = self.jac.tocsc()

    def reset_deriv(self):
        super().reset_deriv()
        self.hess_rows = None
        self.deriv = None

    def compute_deriv(self, active_set):
        inactive_indices = np.where(np.logical_not(self.active_set))[0]
        lamb = 1.0 / self.dt
        rho = self.rho

        m = self.m

        inactive_jac = self.jac[:, inactive_indices]
        inactive_hess = self.hess_rows[:, inactive_indices]

        lower_mat = sp.sparse.diags([-lamb / (1.0 + lamb * rho)], shape=(m, m))

        deriv = sp.sparse.bmat(
            [
                [inactive_hess, inactive_jac.T],
                [inactive_jac, lower_mat],
            ],
            format="csc",
        )

        assert np.allclose((deriv - deriv.T).data, 0.0)

        return deriv

    def compute_rhs(self, active_indices, b0, b1, b2t):
        active_hess = self.hess_rows[:, active_indices]
        active_jac = self.jac[:, active_indices]

        b1c = b1 - (active_hess @ b0)
        b2c = b2t - (active_jac @ b0)

        rhs = np.concatenate((b1c, b2c))

        return rhs

    def solve_scaled(self, b0, b1, b2t):
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

        dx = np.zeros((n,))

        dx[inactive_indices] = inactive_dx
        dx[active_indices] = b0

        return (dx, dy)

    def solve_deriv(self, active_set, deriv, rhs):
        if self.solver is None:
            self.solver = self.linear_solver(self.deriv)

        return self.solver.solve(rhs)

    def solve_active_set(self, active_set, rhs):
        if self.deriv is None:
            self.deriv = self.compute_deriv(self.active_set)

        return self.solve_deriv(self.active_set, self.deriv, rhs)
