import scipy as sp

from pygradflow.params import LinearSolverType

from .linear_solver import LinearSolver, LinearSolverError


def linear_solver(
                mat: sp.sparse.spmatrix, solver_type: LinearSolverType
) -> LinearSolver:
    if solver_type == LinearSolverType.LU:
        from .lu_solver import LUSolver
        return LUSolver(mat)
    elif solver_type == LinearSolverType.MINRES:
        from .minres_solver import MINRESSolver
        return MINRESSolver(mat)
    elif solver_type == LinearSolverType.Cholesky:
        from .cholesky_solver import CholeskySolver
        return CholeskySolver(mat)
    elif solver_type == LinearSolverType.MA57:
        from .ma57_solver import MA57Solver
        return MA57Solver(mat)
    elif solver_type == LinearSolverType.MUMPS:
        from .mumps_solver import MUMPSSolver
        return MUMPSSolver(mat)
    else:
        from .gmres_solver import GMRESSolver
        assert solver_type == LinearSolverType.GMRES
        return GMRESSolver(mat)
