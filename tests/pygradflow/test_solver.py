import numpy as np
import scipy as sp

import pytest

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.problem import Problem
from pygradflow.solver import Solver, SolverStatus
from pygradflow.newton import newton_method

from pygradflow.params import (
    NewtonType,
    Params,
    DerivCheck,
    PenaltyUpdate,
    Precision,
    StepSolverType,
    StepControlType,
    LinearSolverType,
)

from .hs71 import HS71
from .tame import Tame
from .rosenbrock import Rosenbrock

rho = 1.0

newton_types = list(NewtonType)
linear_solver_types = [LinearSolverType.LU, LinearSolverType.GMRES]
step_solver_types = list(StepSolverType)
step_control_types = list(StepControlType)


@pytest.fixture
def rosenbrock_instance():
    problem = Rosenbrock()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([])

    return (problem, x_0, y_0)


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

    next_iterate = newton.step(iterate).iterate

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


@pytest.mark.parametrize("step_control_type", step_control_types)
def test_step_control(hs71_instance, step_control_type):
    problem, x_0, y_0 = hs71_instance
    params = Params(step_control_type=step_control_type)
    solver = Solver(problem, params)

    result = solver.solve(x_0, y_0)

    assert result.success


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
                    report_rcond=True,
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

    next_iterate = method.step(iterate).iterate
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
    "newton_type", [NewtonType.ActiveSet, NewtonType.Simplified, NewtonType.Full]
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


def test_detect_unbounded():
    num_vars = 1

    class UnboundedProblem(Problem):
        def __init__(self):
            var_lb = np.full(shape=(num_vars,), fill_value=-np.inf)
            var_ub = np.full(shape=(num_vars,), fill_value=np.inf)
            super().__init__(var_lb, var_ub)

        def obj(self, x):
            return x[0]

        def obj_grad(self, x):
            return np.array([1.] + [0.] * (num_vars - 1))

        def cons(self, x):
            return np.array([])

        def cons_jac(self, x):
            return sp.sparse.coo_matrix((0, num_vars))

        def lag_hess(self, x, lag):
            return sp.sparse.diags([0.]*num_vars)

    problem = UnboundedProblem()

    solver = Solver(problem)

    x0 = np.array([0.0]*num_vars)
    y0 = np.array([])

    result = solver.solve(x0, y0)

    assert result.status == SolverStatus.Unbounded
