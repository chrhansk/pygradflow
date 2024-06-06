import numpy as np
import scipy as sp

from pygradflow.iterate import Iterate
from pygradflow.linear_solver import LinearSolverError
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.step_solver_error import StepSolverError

from .scaled_step_solver import ScaledStepSolver


class ExtendedStepSolver(ScaledStepSolver):
    """
    Extended step solver. Computes Newton step based on the
    scaled implicit function and a modified formulation in
    order to improve condition and sparsity.
    """

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

    def extract_rows(
        self, mat: sp.sparse.spmatrix, row_filter: np.ndarray
    ) -> sp.sparse.spmatrix:
        mat = mat.tocsc()
        return mat[row_filter, :]

    def _compute_deriv(self) -> None:
        active_indices = np.where(self.active_set)[0]
        inactive_indices = np.where(np.logical_not(self.active_set))[0]

        assert self.dt > 0.0

        lamb = 1.0 / self.dt
        rho = self.rho

        active_set_size = active_indices.size

        n = self.n
        m = self.m

        active_rows = np.arange(active_set_size)
        active_data = np.ones((active_set_size,), dtype=self.params.dtype)

        trans_active_mat = sp.sparse.coo_matrix(
            (active_data, (active_rows, active_indices)), shape=(active_set_size, n)
        )

        lower_mat = sp.sparse.diags(
            [-lamb / (1.0 + lamb * rho)], shape=(m, m), dtype=self.params.dtype
        )

        jac = self.jac
        hess = self.hess + sp.sparse.diags(
            [lamb], shape=(n, n), dtype=self.params.dtype
        )

        filtered_hess = self.extract_rows(hess, inactive_indices)
        trans_filtered_jac = self.extract_rows(jac.T, inactive_indices)

        blocks = [
            [trans_active_mat, None],
            [filtered_hess, trans_filtered_jac],
            [jac, lower_mat],
        ]

        self._deriv = sp.sparse.bmat(blocks, format="csc")

        assert self.deriv.shape == (n + m, n + m)
        assert self.deriv.dtype == self.params.dtype

    def solve_scaled(self, b0, b1, b2t):
        params = self.params
        if self._deriv is None:
            self._compute_deriv()

        n = self.n
        m = self.m

        rhs = np.concatenate((b0, b1, b2t))

        assert rhs.shape == (n + m,)

        try:
            if self.solver is None:
                self.solver = self.linear_solver(self.deriv)
            sol = self.solver.solve(rhs)
        except LinearSolverError as e:
            raise StepSolverError from e

        dx = sol[:n]
        dy = sol[n:]

        rcond = None
        if params.report_rcond:
            try:
                rcond = self.estimate_rcond(self.deriv, self.solver)
            except LinearSolverError:
                pass

        return (dx, dy, rcond)
