from pygradflow.params import Params, StepSolverType
from pygradflow.problem import Problem

from pygradflow.step.step_solver import StepSolver
from pygradflow.step.standard_step_solver import StandardStepSolver
from pygradflow.step.extended_step_solver import ExtendedStepSolver
from pygradflow.step.symmetric_step_solver import SymmetricStepSolver

from pygradflow.iterate import Iterate


def step_solver(
    problem: Problem, params: Params, iterate: Iterate, dt: float, rho: float
) -> StepSolver:
    assert dt > 0.0
    assert rho > 0.0

    if params.step_solver is not None:
        return params.step_solver(problem, params, iterate, dt, rho)

    step_solver_type = params.step_solver_type

    if step_solver_type == StepSolverType.Standard:
        return StandardStepSolver(problem, params, iterate, dt, rho)
    elif step_solver_type == StepSolverType.Extended:
        return ExtendedStepSolver(problem, params, iterate, dt, rho)
    else:
        assert step_solver_type == StepSolverType.Symmetric

        return SymmetricStepSolver(problem, params, iterate, dt, rho)
