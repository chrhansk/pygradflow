import numpy as np
import pytest

from pygradflow.params import LinearSolverType, Params, StepSolverType
from pygradflow.solver import Solver


@pytest.fixture
def target_problem():
    from .target_problem import TargetProblem

    return TargetProblem()


@pytest.fixture
def x0():
    return np.array([2.8, 2.9])


@pytest.fixture
def optima():
    from .target_problem import optima

    return optima


def test_target_cholesky(target_problem, x0, optima):
    pytest.importorskip("pygradflow.linear_solver.cholesky_solver")

    problem = target_problem

    params = Params(
        step_solver_type=StepSolverType.Symmetric,
        linear_solver_type=LinearSolverType.Cholesky,
    )
    solver = Solver(problem, params)

    result = solver.solve(x0)

    assert result.success

    assert any([np.isclose(result.x, opt).all() for opt in optima])


def test_target_MA57(target_problem, x0, optima):
    pytest.importorskip("pygradflow.linear_solver.ma57_solver")

    problem = target_problem

    params = Params(
        step_solver_type=StepSolverType.Symmetric,
        linear_solver_type=LinearSolverType.MA57,
        inertia_correction=True,
    )
    solver = Solver(problem, params)

    result = solver.solve(x0)

    assert result.success

    assert any([np.isclose(result.x, opt).all() for opt in optima])
