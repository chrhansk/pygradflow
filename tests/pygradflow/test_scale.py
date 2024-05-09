import numpy as np
import pytest

from pygradflow.deriv_check import deriv_check
from pygradflow.iterate import Iterate
from pygradflow.params import DerivCheck, Params, ScalingType
from pygradflow.scale import ScaledProblem, Scaling
from pygradflow.solver import Solver

from .instances import hs71_constrained_instance, hs71_instance


def zero_scaling(problem):
    num_vars = problem.num_vars
    num_cons = problem.num_cons

    return Scaling(np.zeros(num_vars, dtype=int), np.zeros(num_cons, dtype=int), 0)


def test_zero_scaled_hs71(hs71_constrained_instance):
    instance = hs71_constrained_instance

    problem = instance.problem
    x_0 = instance.x_0
    y_0 = instance.y_0

    scaling = zero_scaling(problem)

    scaled_problem = ScaledProblem(problem, scaling)

    assert np.allclose(scaled_problem.obj(x_0), problem.obj(x_0))
    assert np.allclose(scaled_problem.obj_grad(x_0), problem.obj_grad(x_0))
    assert np.allclose(scaled_problem.cons(x_0), problem.cons(x_0))

    assert np.allclose(
        scaled_problem.cons_jac(x_0).toarray(), problem.cons_jac(x_0).toarray()
    )
    assert np.allclose(
        scaled_problem.lag_hess(x_0, y_0).toarray(),
        problem.lag_hess(x_0, y_0).toarray(),
    )

    assert np.allclose(scaled_problem.var_lb, problem.var_lb)
    assert np.allclose(scaled_problem.var_ub, problem.var_ub)

    assert np.allclose(scaled_problem.cons_lb, problem.cons_lb)
    assert np.allclose(scaled_problem.cons_ub, problem.cons_ub)


@pytest.fixture
def hs71_scaling():
    obj_weight = 4

    scaling = Scaling(
        np.array([1, 2, 3, -4, 2], dtype=int), np.array([3, 2], dtype=int), obj_weight
    )

    return scaling


def test_objective_weight(hs71_instance, hs71_scaling):
    instance = hs71_instance

    problem = instance.problem
    x_0 = instance.x_0

    scaling = hs71_scaling

    scaled_problem = ScaledProblem(problem, scaling)

    scaled_x_0 = scaling.scale_primal(x_0)

    assert np.allclose(
        scaled_problem.obj(scaled_x_0), problem.obj(x_0) * (2**scaling.obj_weight)
    )


def test_derivs(hs71_instance, hs71_scaling):
    instance = hs71_instance

    problem = instance.problem

    scaling = hs71_scaling

    x_0 = instance.x_0
    y_0 = instance.y_0

    scaled_x_0 = scaling.scale_primal(x_0)
    scaled_y_0 = scaling.scale_dual(y_0)

    params = Params()

    scaled_problem = ScaledProblem(problem, scaling)

    scaled_grad = scaled_problem.obj_grad(scaled_x_0)

    def obj(x):
        return scaled_problem.obj(x)

    deriv_check(obj, scaled_x_0, scaled_grad, params)

    scaled_cons_jac = scaled_problem.cons_jac(scaled_x_0)

    def cons(x):
        return scaled_problem.cons(x)

    deriv_check(cons, scaled_x_0, scaled_cons_jac, params)

    scaled_lag_hess = scaled_problem.lag_hess(scaled_x_0, scaled_y_0)

    def lag_deriv(x):
        obj_grad = scaled_problem.obj_grad(x)
        cons_jac = scaled_problem.cons_jac(x)
        return obj_grad + cons_jac.T.dot(scaled_y_0)

    deriv_check(lag_deriv, scaled_x_0, scaled_lag_hess, params)


def test_scale_inverse(hs71_instance, hs71_scaling):
    instance = hs71_instance

    problem = instance.problem

    scaling = hs71_scaling

    x = instance.x_opt
    y = instance.y_opt

    scaled_x = scaling.scale_primal(x)
    scaled_y = scaling.scale_dual(y)

    unscaled_x = scaling.unscale_primal(scaled_x)
    unscaled_y = scaling.unscale_dual(scaled_y)

    assert (x == unscaled_x).all()
    assert (y == unscaled_y).all()


def test_nominal():
    values = [1e-4, 0.1, 1.0, 100.0, 90.0, -100.0, 1024.0, 1e5]
    weights = Scaling.weights_from_nominal_values(values)

    scaled_values = values * (2.0**weights)
    scaled_values = np.absolute(scaled_values)

    assert (scaled_values >= 1.0).all()
    assert (scaled_values < 2.0).all()


def test_residuals(hs71_instance, hs71_scaling):
    instance = hs71_instance

    problem = instance.problem
    scaling = hs71_scaling

    x_opt = instance.x_opt
    y_opt = instance.y_opt

    params = Params()

    scaled_problem = ScaledProblem(problem, scaling)

    opt_iterate = Iterate(problem, params, x_opt, y_opt)

    assert np.allclose(opt_iterate.stat_res, 0.0, atol=1e-6)
    assert np.allclose(opt_iterate.cons_violation, 0.0, atol=1e-6)

    scaled_x_opt = scaling.scale_primal(x_opt)
    scaled_y_opt = scaling.scale_dual(y_opt)

    scaled_opt_iterate = Iterate(scaled_problem, params, scaled_x_opt, scaled_y_opt)

    bounds_dual = opt_iterate.bounds_dual
    scaled_bounds_dual = scaled_opt_iterate.bounds_dual

    assert np.allclose(bounds_dual, scaling.unscale_bounds_dual(scaled_bounds_dual))
    assert np.allclose(scaled_bounds_dual, scaling.scale_bounds_dual(bounds_dual))

    assert np.allclose(scaled_opt_iterate.stat_res, 0.0, atol=1e-5)
    assert np.allclose(scaled_opt_iterate.cons_violation, 0.0, atol=1e-5)


def test_solve_hs71_scaled(hs71_instance):
    instance = hs71_instance

    problem = instance.problem
    x_0 = instance.x_0
    y_0 = instance.y_0

    scaling = Scaling(
        np.array([1, 2, 3, 2, 1], dtype=int), np.array([2, 3], dtype=int), 0
    )

    params = Params(
        iteration_limit=5000,
        deriv_check=DerivCheck.CheckAll,
        scaling=scaling,
        scaling_type=ScalingType.Custom,
    )

    solver = Solver(problem, params=params)

    result = solver.solve(x_0, y_0)

    assert result.success

    x_opt = instance.x_opt
    y_opt = instance.y_opt

    assert np.allclose(result.x, x_opt, atol=1e-6)
    assert np.allclose(result.y, y_opt, atol=1e-6)
