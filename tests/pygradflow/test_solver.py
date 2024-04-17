import numpy as np
import pytest

from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.newton import newton_method
from pygradflow.params import (
    DerivCheck,
    LinearSolverType,
    NewtonType,
    Params,
    PenaltyUpdate,
    Precision,
    StepControlType,
    StepSolverType,
)
from pygradflow.solver import Solver

from .hs71_cons import HS71Constrained
from .instances import (
    hs71_constrained_instance,
    hs71_instance,
    rosenbrock_instance,
    tame_instance,
)
from .tame import Tame

rho = 1.0

newton_types = list(NewtonType)
linear_solver_types = [LinearSolverType.LU, LinearSolverType.GMRES]
step_solver_types = list(StepSolverType)
step_control_types = [
    StepControlType.Exact,
    StepControlType.ResiduumRatio,
    StepControlType.DistanceRatio,
]


def test_custom_step_solver(rosenbrock_instance):
    from pygradflow.step.symmetric_step_solver import SymmetricStepSolver

    problem = rosenbrock_instance.problem
    x_0 = rosenbrock_instance.x_0
    y_0 = rosenbrock_instance.y_0

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


def solve_and_test_instance(instance, solver):
    result = solver.solve(instance.x_0, instance.y_0)

    assert result.success

    x_opt = instance.x_opt
    y_opt = instance.y_opt

    assert np.allclose(result.x, x_opt, atol=1e-6)
    assert np.allclose(result.y, y_opt, atol=1e-6)


def test_solve_rosenbrock(rosenbrock_instance):
    problem = rosenbrock_instance.problem

    params = Params(iteration_limit=100)
    solver = Solver(problem, params)

    solve_and_test_instance(rosenbrock_instance, solver)


@pytest.mark.parametrize("step_control_type", step_control_types)
def test_step_control(hs71_instance, step_control_type):
    problem = hs71_instance.problem
    params = Params(step_control_type=step_control_type)
    solver = Solver(problem, params)

    solve_and_test_instance(hs71_instance, solver)


def test_solve_hs71(hs71_instance):
    problem = hs71_instance.problem
    solver = Solver(problem)

    solve_and_test_instance(hs71_instance, solver)


def test_solve_hs71_constrained(hs71_constrained_instance):
    problem = hs71_constrained_instance.problem
    solver = Solver(problem)

    solve_and_test_instance(hs71_constrained_instance, solver)


@pytest.mark.parametrize(
    "penalty_update",
    [PenaltyUpdate.Constant, PenaltyUpdate.DualNorm, PenaltyUpdate.ParetoDecrease],
)
def test_penalty_update(hs71_instance, penalty_update):
    problem = hs71_instance.problem
    params = Params(penalty_update=penalty_update)
    solver = Solver(problem, params)

    solve_and_test_instance(hs71_instance, solver)


@pytest.mark.parametrize("newton_type", newton_types)
@pytest.mark.parametrize("step_solver_type", step_solver_types)
@pytest.mark.parametrize("linear_solver_type", linear_solver_types)
def test_solve_hs71_single(
    hs71_instance, newton_type, step_solver_type, linear_solver_type
):
    problem = hs71_instance.problem
    x_0 = hs71_instance.x_0
    y_0 = hs71_instance.y_0

    params = Params(
        precision=Precision.Single,
        newton_type=newton_type,
        step_solver_type=step_solver_type,
        iteration_limit=10,
        report_rcond=True,
        linear_solver_type=linear_solver_type,
    )

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
def test_one_step_convergence(
    tame_instance, newton_type, step_solver_type, linear_solver_type
):
    problem = tame_instance.problem
    x_0 = tame_instance.x_0
    y_0 = tame_instance.y_0

    params = Params(
        newton_type=newton_type,
        step_solver_type=step_solver_type,
        linear_solver_type=linear_solver_type,
    )

    dt = 10.0

    iterate = Iterate(problem, params, x_0, y_0)
    method = newton_method(problem, params, iterate, dt, rho)

    next_iterate = method.step(iterate).iterate
    func = ImplicitFunc(problem, iterate, dt)

    assert np.allclose(func.value_at(next_iterate, rho=rho), 0.0)


def test_solve_tame(tame_instance):
    problem = tame_instance.problem
    params = Params(newton_type=NewtonType.Full, deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    solve_and_test_instance(tame_instance, solver)


# TODO: Find out why full Newton does not converge
@pytest.mark.parametrize(
    "newton_type", [NewtonType.ActiveSet, NewtonType.Simplified, NewtonType.Full]
)
def test_solve_with_newton_types(hs71_instance, newton_type):
    problem = hs71_instance.problem

    params = Params(
        newton_type=newton_type, rho=1.0, penalty_update=PenaltyUpdate.Constant
    )

    solver = Solver(problem, params)

    solve_and_test_instance(hs71_instance, solver)


def test_grad_errors(tame_instance):
    problem = tame_instance.problem

    orig_obj_grad = problem.obj_grad

    def obj_grad(x):
        g = orig_obj_grad(x)
        g[0] += 1.0
        return g

    problem.obj_grad = obj_grad

    x_0 = tame_instance.x_0
    y_0 = tame_instance.y_0

    params = Params(deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    with pytest.raises(ValueError) as e:
        solver.solve(x_0, y_0)

    e = e.value
    assert (e.invalid_indices == [[0, 0]]).all()


def test_cons_errors(tame_instance):
    problem = tame_instance.problem

    invalid_index = 1

    orig_cons_jac = problem.cons_jac

    def cons_jac(x):
        g = orig_cons_jac(x)
        g.data[invalid_index] += 1.0
        return g

    problem.cons_jac = cons_jac

    x_0 = tame_instance.x_0
    y_0 = tame_instance.y_0

    params = Params(deriv_check=DerivCheck.CheckAll)

    solver = Solver(problem, params)

    with pytest.raises(ValueError) as e:
        solver.solve(x_0, y_0)

    e = e.value
    jac = Tame().cons_jac(x_0)

    invalid_row = jac.row[invalid_index]
    invalid_col = jac.col[invalid_index]

    assert e.invalid_indices == [invalid_row]
    assert e.col_index == invalid_col
