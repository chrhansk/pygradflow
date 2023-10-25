import numpy as np

import pytest
import scipy as sp

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.solver import Solver
from pygradflow.newton import newton_method

from pygradflow.params import (
    NewtonType,
    Params,
    DerivCheck,
    PenaltyUpdate,
    Precision,
    StepSolverType,
    LinearSolverType,
)

from .hs71 import HS71
from .tame import Tame
from .rosenbrock import Rosenbrock

rho = 1.0

newton_types = [e for e in NewtonType]
linear_solver_types = [LinearSolverType.LU, LinearSolverType.GMRES]
step_solver_types = [e for e in StepSolverType]


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

    next_iterate = newton.step(iterate)

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

    next_iterate = newton.step(iterate)

    xclip = np.clip(x_0, problem.var_lb, problem.var_ub)

    assert np.allclose(next_iterate.x, xclip)
    assert np.allclose(next_iterate.y, iterate.y)


def test_custom_step_solver(rosenbrock_instance):
    from pygradflow.step.symmetric_step_solver import SymmetricStepSolver

    problem, x_0, y_0 = rosenbrock_instance

    params = Params(newton_type=NewtonType.Full, step_solver=SymmetricStepSolver)

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

    next_iterate = newton.step(iterate)

    xclip = np.clip(x_0, problem.var_lb, problem.var_ub)

    assert np.allclose(next_iterate.x, xclip)
    assert np.allclose(next_iterate.y, iterate.y)


def test_solve_rosenbrock(rosenbrock_instance):
    problem, x_0, y_0 = rosenbrock_instance
    params = Params(num_it=100)

    solver = Solver(problem, params)

    result = solver.solve(x_0, y_0)

    assert result.success


@pytest.fixture
def hs71_instance():
    problem = HS71()

    x_0 = np.array([1.0, 5.0, 5.0, 1.0, 0.0])
    y_0 = np.array([0.0, 0.0])

    return (problem, x_0, y_0)


def test_solve_hs71(hs71_instance):
    problem, x_0, y_0 = hs71_instance
    solver = Solver(problem)

    result = solver.solve(x_0, y_0)

    assert result.success


@pytest.mark.parametrize("newton_type", newton_types)
@pytest.mark.parametrize("step_solver_type", step_solver_types)
@pytest.mark.parametrize("linear_solver_type", linear_solver_types)
def test_solve_hs71_single(hs71_instance,
                           newton_type,
                           step_solver_type,
                           linear_solver_type):
    problem, x_0, y_0 = hs71_instance
    params = Params(precision=Precision.Single,
                    newton_type=newton_type,
                    step_solver_type=step_solver_type,
                    num_it=10,
                    linear_solver_type=linear_solver_type)

    solver = Solver(problem, params)

    result = solver.solve(x_0, y_0)

    # Takes many more iterations to converge
    # assert result.success

    x = result.x
    y = result.y

    assert x.dtype == np.float32
    assert y.dtype == np.float32


@pytest.mark.parametrize("newton_type", newton_types)
@pytest.mark.parametrize("step_solver_type", step_solver_types)
@pytest.mark.parametrize("linear_solver_type", linear_solver_types)
def test_one_step_convergence(newton_type, step_solver_type, linear_solver_type):
    problem = Tame()

    params = Params(
        newton_type=newton_type,
        step_solver_type=step_solver_type,
        linear_solver_type=linear_solver_type,
    )

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])
    dt = 10.0

    iterate = Iterate(problem, params, x_0, y_0)
    method = newton_method(problem, params, iterate, dt, rho)

    next_iterate = method.step(iterate)
    func = ImplicitFunc(problem, iterate, dt)

    assert np.allclose(func.value_at(next_iterate, rho=rho), 0.0)


def test_solve_tame():
    problem = Tame()
    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])
    params = Params(newton_type=NewtonType.Full,
                    deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    result = solver.solve(x_0, y_0)

    assert result.success


# TODO: Find out why full Newton does not converge
@pytest.mark.parametrize(
    "newton_type", [NewtonType.ActiveSet, NewtonType.Simple, NewtonType.Full]
)
def test_solve_with_newton_types(hs71_instance, newton_type):
    problem, x_0, y_0 = hs71_instance

    params = Params(
        newton_type=newton_type, rho=1.0, penalty_update=PenaltyUpdate.Constant
    )

    solver = Solver(problem, params)

    result = solver.solve(x_0, y_0)

    assert result.success


def test_grad_errors():
    problem = Tame()

    def obj_grad(x):
        g = Tame().obj_grad(x)
        g[0] += 1.
        return g

    problem.obj_grad = obj_grad

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])
    params = Params(deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    with pytest.raises(ValueError) as e:
        solver.solve(x_0, y_0)

    e = e.value
    assert (e.invalid_indices == [[0, 0]]).all()


def test_cons_errors():
    problem = Tame()

    invalid_index = 1

    def cons_jac(x):
        g = Tame().cons_jac(x)
        g.data[invalid_index] += 1.
        return g

    problem.cons_jac = cons_jac

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])
    params = Params(deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    with pytest.raises(ValueError) as e:
        solver.solve(x_0, y_0)

    e = e.value
    jac = Tame().cons_jac(x_0)

    invalid_row = jac.row[invalid_index]
    invalid_col = jac.col[invalid_index]

    assert (e.invalid_indices == [[invalid_row, invalid_col]]).all()
