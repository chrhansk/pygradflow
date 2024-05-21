import numpy as np
import pytest
import scipy as sp

from pygradflow.integration.integration_solver import IntegrationSolver
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.status import SolverStatus

from ..instances import hs71_instance, tame_instance


@pytest.fixture
def integration_params():
    return Params(iteration_limit=1000, rho=1e-2)


class SimpleProblem(Problem):
    def __init__(self):
        var_lb = np.array([-np.inf])
        var_ub = np.array([np.inf])
        super().__init__(var_lb, var_ub, num_cons=0)

    def obj(self, x):
        return 0.5 * x[0] ** 2

    def obj_grad(self, x):
        return np.array([x[0]])

    def lag_hess(self, x, y):
        return sp.sparse.eye(1)


def test_simple_problem(integration_params):
    problem = SimpleProblem()
    x0 = np.array([10.0])
    y0 = np.array([])

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert result.status == SolverStatus.Optimal


class SimpleUnboundedProblem(Problem):
    def __init__(self):
        var_lb = np.array([-np.inf])
        var_ub = np.array([np.inf])
        super().__init__(var_lb, var_ub, num_cons=0)

    def obj(self, x):
        return x[0]

    def obj_grad(self, x):
        return np.array([1.0])

    def lag_hess(self, x, y):
        return sp.sparse.csr_matrix((1, 1))


def test_simple_unbounded(integration_params):
    problem = SimpleUnboundedProblem()
    x0 = np.array([0.0])
    y0 = np.array([])

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert result.status == SolverStatus.Unbounded


class ActiveSetChangeProblem(Problem):
    def __init__(self):
        var_lb = np.array([1.0])
        var_ub = np.array([np.inf])
        super().__init__(var_lb, var_ub, num_cons=0)

    def obj(self, x):
        return 0.5 * x[0] ** 2

    def obj_grad(self, x):
        return np.array([x[0]])

    def lag_hess(self, x, y):
        return sp.sparse.eye(1)


def test_solve_active_set_change(integration_params):
    problem = ActiveSetChangeProblem()
    x0 = np.array([10.0])
    y0 = np.array([])

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert (result.x == 1.0).all()

    assert result.success


class SingleActiveSetProblem(Problem):
    def __init__(self):
        var_lb = np.array([1.0, -np.inf])
        var_ub = np.array([np.inf, np.inf])
        super().__init__(var_lb, var_ub, num_cons=0)

    def obj(self, z):
        return 0.5 * np.dot(z, z)

    def obj_grad(self, z):
        return z

    def lag_hess(self, z, y):
        return sp.sparse.eye(2)


def test_solve_single_active_set(integration_params):
    problem = SingleActiveSetProblem()
    x0 = np.array([1.5, 10.0])
    y0 = np.array([])

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert np.allclose(result.x, np.array([1.0, 0.0]), atol=1e-6)

    assert result.success


def test_solve_tame(tame_instance, integration_params):
    problem = tame_instance.problem
    x0 = tame_instance.x_0
    y0 = tame_instance.y_0

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert result.success

    assert np.allclose(result.x, tame_instance.x_opt, atol=1e-6)
    assert np.allclose(result.y, tame_instance.y_opt, atol=1e-6)


def test_solve_hs71(hs71_instance, integration_params):
    problem = hs71_instance.problem
    x0 = hs71_instance.x_0
    y0 = hs71_instance.y_0

    solver = IntegrationSolver(problem, integration_params)

    result = solver.solve(x0, y0)

    assert result.success

    assert np.allclose(result.x, hs71_instance.x_opt, atol=1e-6)
    assert np.allclose(result.y, hs71_instance.y_opt, atol=1e-6)
