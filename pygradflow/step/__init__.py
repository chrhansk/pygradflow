from pygradflow.params import StepSolverType
from pygradflow.step.extended_step_solver import ExtendedStepSolver
from pygradflow.step.standard_step_solver import StandardStepSolver
from pygradflow.step.symmetric_step_solver import SymmetricStepSolver


def step_solver(problem, params, iterate, dt, rho):
    assert dt > 0.0
    assert rho > 0.0

    step_solver_type = params.step_solver_type

    if step_solver_type == StepSolverType.Standard:
        return StandardStepSolver(problem, params, iterate, dt, rho)
    elif step_solver_type == StepSolverType.Extended:
        return ExtendedStepSolver(problem, params, iterate, dt, rho)
    else:
        assert step_solver_type == StepSolverType.Symmetric

        return SymmetricStepSolver(problem, params, iterate, dt, rho)
