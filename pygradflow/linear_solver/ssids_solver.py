import pyspral.ssids as ssids

from .linear_solver import LinearSolver, LinearSolverError


class SSIDSSolver(LinearSolver):
    def __init__(self, mat, symmetric=False, report_rcond=False):
        super().__init__(mat, symmetric=symmetric)

        try:
            self.symbolic_factor = ssids.analyze(mat, check=True)
            self.numeric_factor = self.symbolic_factor.factor(posdef=False)
        except ssids.SSIDSError as e:
            raise LinearSolverError() from e

    def solve(self, rhs, trans=False, initial_sol=None):
        try:
            return self.numeric_factor.solve(rhs, inplace=False)
        except ssids.SSIDSError as e:
            raise LinearSolverError() from e

    def num_neg_eigvals(self):
        return self.numeric_factor.inform.num_neg
