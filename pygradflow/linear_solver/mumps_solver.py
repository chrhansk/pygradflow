import warnings

import mumps
import numpy as np
import scipy as sp

from .linear_solver import LinearSolver, LinearSolverError


class MUMPSSolver(LinearSolver):
    def __init__(self, mat: sp.sparse.spmatrix, symmetric=False) -> None:
        super().__init__(mat, symmetric=symmetric)

        sym = 2 if symmetric else 0

        self.ctx = mumps.DMumpsContext(sym=sym, par=1)

        self.ctx.set_icntl(13, 1)

        self.mat = mat

        if mat.format != "coo":
            warnings.warn(
                "Converting matrix to COO format", sp.sparse.SparseEfficiencyWarning
            )
            self.mat = mat.tocoo()

        rows = self.mat.row
        cols = self.mat.col
        data = self.mat.data

        if symmetric:
            filter = rows >= cols
            rows = rows[filter]
            cols = cols[filter]
            data = data[filter]

        if data.dtype != np.float64:
            warnings.warn(
                "Converting matrix data to float64", sp.sparse.SparseEfficiencyWarning
            )
            data = data.astype(np.float64)

        self.ctx.set_shape(self.mat.shape[0])
        self.ctx.set_centralized_assembled(rows + 1, cols + 1, data)

        # Analysis
        self.ctx.run(job=1)

        # Factorization
        self.ctx.run(job=2)

    def solve(self, rhs, trans=False, initial_sol=None):
        sol = np.copy(rhs)

        if sol.dtype != np.float64:
            warnings.warn(
                "Converting rhs to float64", sp.sparse.SparseEfficiencyWarning
            )
            sol = sol.astype(np.float64)

        if not self.symmetric:
            if trans:
                self.ctx.set_icntl(9, 0)
            else:
                self.ctx.set_icntl(9, 1)

        self.ctx.set_rhs(sol)

        try:
            # Solution
            self.ctx.run(job=3)
        except RuntimeError as e:
            raise LinearSolverError from e

        return sol.astype(rhs.dtype)

    def num_neg_eigvals(self):
        return self.ctx.id.infog[11]

    def rcond(self):
        return None
