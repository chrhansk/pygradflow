import scipy as sp

from pygradflow.params import LinearSolverType

from .linear_solver import LinearSolver, LinearSolverError


def linear_solver(
    mat: sp.sparse.spmatrix, solver_type: LinearSolverType, symmetric=False
) -> LinearSolver:
    if solver_type == LinearSolverType.LU:
        from .lu_solver import LUSolver

        return LUSolver(mat, symmetric=symmetric)
    elif solver_type == LinearSolverType.MINRES:
        from .minres_solver import MINRESSolver

        return MINRESSolver(mat, symmetric=symmetric)
    elif solver_type == LinearSolverType.Cholesky:
        from .cholesky_solver import CholeskySolver

        return CholeskySolver(mat, symmetric=symmetric)
    elif solver_type == LinearSolverType.MA57:
        from .ma57_solver import MA57Solver

        return MA57Solver(mat, symmetric=symmetric)
    elif solver_type == LinearSolverType.MUMPS:
        from .mumps_solver import MUMPSSolver

        return MUMPSSolver(mat, symmetric=symmetric)
    elif solver_type == LinearSolverType.SSIDS:
        from .ssids_solver import SSIDSSolver

        return SSIDSSolver(mat, symmetric=symmetric)
    else:
        from .gmres_solver import GMRESSolver

        assert solver_type == LinearSolverType.GMRES
        return GMRESSolver(mat, symmetric=symmetric)


__all__ = ["linear_solver", "LinearSolverError", "LinearSolver"]
