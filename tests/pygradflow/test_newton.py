import numpy as np
import pytest
import scipy as sp

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.newton import newton_method
from pygradflow.params import LinearSolverType, NewtonType, Params, StepSolverType

from .rosenbrock import Rosenbrock

rho = 1.0

newton_types = list(NewtonType)
linear_solver_types = [LinearSolverType.LU, LinearSolverType.GMRES]
step_solver_types = list(StepSolverType)


@pytest.fixture
def rosenbrock_instance():
    problem = Rosenbrock()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([])

    return (problem, x_0, y_0)


def test_newton_active_set(rosenbrock_instance):
    problem, x_0, y_0 = rosenbrock_instance
    params = Params()

    iterate = Iterate(problem, params, x_0, y_0)
    dt = 1e-10

    n = x_0.shape[0]

    problem.var_lb = x_0 + np.ones((n,))
    problem.var_ub = x_0 + np.ones((n,))
    problem.var_ub[-1] = np.inf

    func = ImplicitFunc(problem, iterate, dt)

    active_set = func.compute_active_set(x_0)

    assert (active_set).all()

    problem.var_lb = x_0 - np.ones((n,))
    problem.var_ub = x_0 + np.ones((n,))
    problem.var_ub[-1] = np.inf

    active_set = func.compute_active_set(x_0)

    assert np.logical_not(active_set).all()


def test_newton_active_set_projection(rosenbrock_instance):
    problem, x_0, y_0 = rosenbrock_instance
    params = Params()

    iterate = Iterate(problem, params, x_0, y_0)
    dt = 1e-10

    n = x_0.shape[0]

    problem.var_lb = x_0 - np.ones((n,))
    problem.var_ub = x_0 + np.ones((n,))

    func = ImplicitFunc(problem, iterate, dt)

    active_set = np.full((n,), True, dtype=bool)

    def test_projection_point(x, active_set):
        proj_x = func.project(x, active_set)

        inactive_set = np.logical_not(active_set)

        x_clipped = np.clip(x, problem.var_lb, problem.var_ub)

        assert np.allclose(proj_x[active_set], x_clipped[active_set])
        assert np.allclose(proj_x[inactive_set], x[inactive_set])

    def test_projection(active_set):
        x = problem.var_ub + 1.0
        test_projection_point(x, active_set)

        x = problem.var_lb - 1.0
        test_projection_point(x, active_set)

        x = np.random.uniform(problem.var_lb, problem.var_ub)
        test_projection_point(x, active_set)

    # All rows are active
    active_set[:] = True
    test_projection(active_set)

    # All rows are inactive
    active_set[:] = False
    test_projection(active_set)

    # All rows are inactive except the first
    active_set[:] = False
    active_set[0] = True
    test_projection(active_set)


def test_active_set_project_deriv(rosenbrock_instance):
    problem, x_0, y_0 = rosenbrock_instance
    params = Params()

    iterate = Iterate(problem, params, x_0, y_0)
    dt = 0.0
    func = ImplicitFunc(problem, iterate, dt)

    n = x_0.shape[0]

    active_set = np.full((n,), False, dtype=bool)

    np.random.seed(0)
    mat = sp.sparse.coo_array(np.random.rand(n, n))

    proj_mat = func.apply_project_deriv(mat, active_set)

    assert np.allclose(proj_mat.toarray(), mat.toarray())

    active_set[:] = True

    proj_mat = func.apply_project_deriv(mat, active_set)

    assert np.allclose(proj_mat.toarray(), 0.0)

    active_set[0] = False

    proj_mat = func.apply_project_deriv(mat, active_set)

    # Inactive row is zero
    assert np.allclose(proj_mat.toarray()[1:, :], 0.0)
    # Active rows are unchanged
    assert np.allclose(proj_mat.toarray()[0, :], mat.toarray()[0, :])


@pytest.mark.parametrize("newton_type", newton_types)
@pytest.mark.parametrize("step_solver_type", step_solver_types)
@pytest.mark.parametrize("linear_solver_type", linear_solver_types)
def test_newton_step_unconstrained(
    rosenbrock_instance, newton_type, step_solver_type, linear_solver_type
):
    problem, x_0, y_0 = rosenbrock_instance

    params = Params(
        newton_type=newton_type,
        step_solver_type=step_solver_type,
        linear_solver_type=linear_solver_type,
    )

    iterate = Iterate(problem, params, x_0, y_0)
    dt = 1e-10
    func = ImplicitFunc(problem, iterate, dt)

    newton = newton_method(problem, params, iterate, dt, rho)

    deriv = func.deriv_at(iterate, rho=rho)

    (n,) = x_0.shape
    (m,) = y_0.shape
    s = n + m

    assert np.allclose(deriv.toarray(), np.eye(s))

    next_iterate = newton.step(iterate).iterate

    assert np.allclose(next_iterate.x, iterate.x)
    assert np.allclose(next_iterate.y, iterate.y)


@pytest.mark.parametrize("newton_type", newton_types)
@pytest.mark.parametrize("step_solver_type", step_solver_types)
@pytest.mark.parametrize("linear_solver_type", linear_solver_types)
def test_newton_step_constrained(
    rosenbrock_instance, newton_type, step_solver_type, linear_solver_type
):
    problem, x_0, y_0 = rosenbrock_instance

    params = Params(
        newton_type=newton_type,
        step_solver_type=step_solver_type,
        linear_solver_type=linear_solver_type,
    )

    (n,) = x_0.shape
    (m,) = y_0.shape

    problem.var_lb = x_0 + np.ones((n,))
    problem.var_ub = x_0 + np.ones((n,))
    problem.var_ub[-1] = np.inf

    iterate = Iterate(problem, params, x_0, y_0)
    dt = 1e-12
    func = ImplicitFunc(problem, iterate, dt)

    newton = newton_method(problem, params, iterate, dt, rho)

    deriv = func.deriv_at(iterate, rho=rho)

    s = n + m

    assert np.allclose(deriv.toarray(), np.eye(s))

    next_iterate = newton.step(iterate).iterate

    xclip = np.clip(x_0, problem.var_lb, problem.var_ub)

    assert np.allclose(next_iterate.x, xclip)
    assert np.allclose(next_iterate.y, iterate.y)
