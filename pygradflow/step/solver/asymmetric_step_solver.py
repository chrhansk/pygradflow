import copy

import numpy as np
import scipy as sp

from pygradflow.iterate import Iterate
from pygradflow.linear_solver import LinearSolverError
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.step_solver_error import StepSolverError

from .scaled_step_solver import ScaledStepSolver


class AsymmetricStepSolver(ScaledStepSolver):
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

    def update_derivs(self, iterate: Iterate) -> None:
        self._jac = copy.copy(iterate.aug_lag_deriv_xy())
        self._hess = copy.copy(iterate.aug_lag_deriv_xx(rho=0.0))
        self.reset_deriv()

    def update_active_set(self, active_set: np.ndarray) -> None:
        self._active_set = copy.copy(active_set)
        self.reset_deriv()

    def overwrite_active_rows(self, matrix):
        m = self.m
        n = self.n

        active_set = self.active_set

        assert matrix.shape == (n + m, n + m)

        data = matrix.data
        cols = matrix.indices
        col_index = matrix.indptr

        for j in range(n):
            active = active_set[j]

            if not active:
                continue

            col_start = col_index[j]
            col_end = col_index[j + 1]

            curr_data = data[col_start:col_end]
            curr_cols = cols[col_start:col_end]

            assert curr_data.shape == curr_cols.shape

            (col_nnz,) = curr_data.shape

            k = np.searchsorted(curr_cols, j)

            curr_data[:] = 0.0
            curr_data[k] = 1.0
            curr_cols[k] = j

            assert (curr_cols[:-1] <= curr_cols[1:]).all()
            assert (0 <= curr_cols).all()
            assert (curr_cols < n + m).all()

    def compute_deriv(self, active_set: np.ndarray) -> sp.sparse.spmatrix:
        lamb = 1.0 / self.dt
        rho = self.rho

        m = self.m
        n = self.n

        jac = self.jac
        hess = self.hess

        hess += sp.sparse.diags([lamb], shape=(n, n), dtype=self.params.dtype)

        lower_mat = sp.sparse.diags(
            [-lamb / (1.0 + lamb * rho)], shape=(m, m), dtype=self.params.dtype
        )

        deriv = sp.sparse.bmat(
            [
                [hess, jac.T],
                [jac, lower_mat],
            ],
            format="csr",
        )

        self.overwrite_active_rows(deriv)

        assert deriv.dtype == self.params.dtype

        return deriv

    def compute_rhs(self, b0, b1, b2t):
        m = self.m
        n = self.n

        params = self.params
        active_set = self.active_set

        rhs = np.empty((n + m,), dtype=params.dtype)

        if m > 0:
            rhs[-m:] = b2t

        var_rhs = rhs[:n]

        var_rhs[active_set] = b0
        var_rhs[np.logical_not(active_set)] = b1

        return rhs

    def initial_sol(self, b0, b1, b2t):
        m = self.m
        n = self.n

        params = self.params
        active_set = self.active_set

        def initial_sol():
            sol = np.zeros((n + m,), dtype=params.dtype)
            var_sol = sol[:n]
            var_sol[active_set] = b0
            return sol

        return initial_sol

    def solve_scaled(self, b0, b1, b2t):
        if self._deriv is None:
            self._deriv = self.compute_deriv(self.active_set)

        if self.solver is None:
            self.solver = self.linear_solver(self.deriv)

        params = self.params

        rhs = self.compute_rhs(b0, b1, b2t)

        initial_sol = self.initial_sol(b0, b1, b2t)

        try:
            sol = self.solver.solve(rhs, initial_sol=initial_sol)
        except LinearSolverError as e:
            raise StepSolverError from e

        m = self.m
        n = self.n

        assert sol.shape == (m + n,)

        var_sol = sol[:n]
        cons_sol = sol[n:]

        rcond = None
        if params.report_rcond:
            try:
                rcond = self.estimate_rcond(self.deriv, self.solver)
            except LinearSolverError:
                pass

        return (var_sol, cons_sol, rcond)
