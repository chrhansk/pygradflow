from typing import Optional

import numpy as np
import scipy as sp

from pygradflow.params import Params, ScalingType
from pygradflow.problem import Problem
from pygradflow.util import sparse_zero


def scale_symmetric(A):
    (n, _) = A.shape

    A = A.tocoo()
    a_rows = A.row
    a_cols = A.col
    a_data = np.abs(A.data)

    max_it = 100

    D = np.zeros((n,), dtype=int)

    for i in range(max_it):
        R = np.zeros((n,), dtype=int)

        for k in range(len(a_data)):
            R[a_cols[k]] += a_data[k]

        R[R < 1e-10] = 1.0
        R = np.sqrt(R)

        Rsca = 1 - np.frexp(R)[1]

        if (Rsca == 0).all():
            break

        for k in range(len(a_data)):
            a_data[k] = np.ldexp(a_data[k], Rsca[a_rows[k]] + Rsca[a_cols[k]])

        D += Rsca
    else:
        raise Exception("Equilibration failed to converge")

    return D


class Scaling:
    def __init__(self, var_weights, cons_weights, obj_weight=0):
        self.var_weights = var_weights
        self.cons_weights = cons_weights

        assert var_weights.ndim == 1
        assert var_weights.dtype in [np.int64, np.int32, np.int16, np.int8]

        assert cons_weights.ndim == 1
        assert cons_weights.dtype in [np.int64, np.int32, np.int16, np.int8]

        self.obj_weight = obj_weight

    @staticmethod
    def zero(num_vars, num_cons):
        return Scaling(
            np.zeros((num_vars,), dtype=int), np.zeros((num_cons,), dtype=int)
        )

    @staticmethod
    def from_nominal_values(var_values, cons_values, obj_value=1.0):
        var_weights = Scaling.weights_from_nominal_values(var_values)
        cons_weights = Scaling.weights_from_nominal_values(cons_values)
        obj_weight = Scaling.weights_from_nominal_values(obj_value)

        return Scaling(var_weights, cons_weights, obj_weight)

    @staticmethod
    def weights_from_nominal_values(values):
        return 1 - np.frexp(values)[1]

    @staticmethod
    def from_grad_jac(obj_grad, cons_jac):
        grad_weights = Scaling.weights_from_nominal_values(np.abs(obj_grad))
        var_weights = -grad_weights

        if cons_jac is None:
            cons_weights = np.zeros((0,), dtype=int)
            return Scaling(var_weights, cons_weights)

        (num_cons, num_vars) = cons_jac.shape
        assert obj_grad.shape == (num_vars,)

        jac = cons_jac.tocoo()

        rows = jac.row
        cols = jac.col
        data = np.abs(jac.data)

        prescaled_data = np.ldexp(data, -var_weights[cols])
        max_values = np.zeros((num_cons,), dtype=int)

        for i, row in enumerate(rows):
            max_values[row] = max(max_values[row], prescaled_data[i])

        cons_weights = Scaling.weights_from_nominal_values(max_values)

        return Scaling(var_weights, cons_weights)

    @staticmethod
    def from_equilibrated_kkt(lag_hess, cons_jac):
        (m, n) = cons_jac.shape
        assert lag_hess.shape == (n, n)

        kkt_mat = sp.sparse.bmat([[lag_hess, cons_jac.T], [cons_jac, None]])

        weights = scale_symmetric(kkt_mat)

        var_weights = -weights[:n]
        cons_weights = weights[n:]

        return Scaling(var_weights, cons_weights)

    @property
    def num_vars(self):
        return len(self.var_weights)

    @property
    def num_cons(self):
        return len(self.cons_weights)

    def scale_primal(self, x):
        return np.ldexp(x, self.var_weights)

    def unscale_primal(self, x):
        return np.ldexp(x, -self.var_weights)

    def _dual_weights(self):
        return self.cons_weights - self.obj_weight

    def _bound_weights(self):
        return self.var_weights - self.obj_weight

    def scale_dual(self, y):
        return np.ldexp(y, -self._dual_weights())

    def unscale_dual(self, y):
        return np.ldexp(y, self._dual_weights())

    def scale_bounds_dual(self, y):
        return np.ldexp(y, -self._bound_weights())

    def unscale_bounds_dual(self, y):
        return np.ldexp(y, self._bound_weights())


class ScaledProblem(Problem):
    def __init__(self, problem: Problem, scaling: Scaling):
        self.problem = problem
        self.scaling = scaling

        var_lb = np.ldexp(problem.var_lb, scaling.var_weights)
        var_ub = np.ldexp(problem.var_ub, scaling.var_weights)

        cons_lb = np.ldexp(problem.cons_lb, scaling.cons_weights)
        cons_ub = np.ldexp(problem.cons_ub, scaling.cons_weights)

        super().__init__(var_lb, var_ub, cons_lb=cons_lb, cons_ub=cons_ub)

    def _orig_x(self, scaled_x):
        var_weights = self.scaling.var_weights
        return np.ldexp(scaled_x, -var_weights)

    def obj(self, x):
        x_orig = self._orig_x(x)
        obj_orig = self.problem.obj(x_orig)
        return np.ldexp(obj_orig, self.scaling.obj_weight)

    def obj_grad(self, x):
        var_weights = self.scaling.var_weights
        x_orig = self._orig_x(x)
        grad_orig = self.problem.obj_grad(x_orig)
        grad = np.ldexp(grad_orig, -var_weights)

        return np.ldexp(grad, self.scaling.obj_weight)

    def cons(self, x):
        cons_weights = self.scaling.cons_weights
        x_orig = self._orig_x(x)

        cons_orig = self.problem.cons(x_orig)

        return np.ldexp(cons_orig, cons_weights)

    def cons_jac(self, x):
        var_weights = self.scaling.var_weights
        cons_weights = self.scaling.cons_weights

        x_orig = self._orig_x(x)

        jac_orig = self.problem.cons_jac(x_orig)

        jac = jac_orig.tocoo()

        jac_row = jac.row
        jac_col = jac.col
        jac_data = jac.data

        for k, (i, j, v) in enumerate(zip(jac_row, jac_col, jac_data)):
            jac_data[k] = np.ldexp(v, cons_weights[i] - var_weights[j])

        return jac

    def lag_hess(self, x, y):
        var_weights = self.scaling.var_weights
        cons_weights = self.scaling.cons_weights
        obj_weight = self.scaling.obj_weight

        x_orig = self._orig_x(x)
        y_orig = np.ldexp(y, cons_weights - obj_weight)

        hess_orig = self.problem.lag_hess(x_orig, y_orig)

        hess = hess_orig.tocoo()

        hess_row = hess.row
        hess_col = hess.col
        hess_data = hess.data

        for k, (i, j, v) in enumerate(zip(hess_row, hess_col, hess_data)):
            combined_weight = obj_weight - var_weights[i] - var_weights[j]
            hess_data[k] = np.ldexp(v, combined_weight)

        return hess


def create_scaling(
    problem: Problem, params: Params, x0: np.ndarray, y0: np.ndarray
) -> Optional[Scaling]:
    scaling_type = params.scaling_type

    if params.scaling is not None:
        assert scaling_type == ScalingType.Custom
        return params.scaling

    if scaling_type == ScalingType.NoScaling:
        return None
    elif scaling_type == ScalingType.Custom:
        raise ValueError("Custom scaling requires explicit scaling")

    if scaling_type == ScalingType.Nominal:
        if problem.num_cons > 0:
            cons_val = problem.cons(x0)
        else:
            cons_val = np.array([], dtype=x0.dtype)
        return Scaling.from_nominal_values(x0, cons_val)

    if problem.num_cons > 0:
        cons_jac = problem.cons_jac(x0)
    else:
        cons_jac = sparse_zero(shape=(0, problem.num_vars))

    if scaling_type == ScalingType.GradJac:
        obj_grad = problem.obj_grad(x0)
        return Scaling.from_grad_jac(obj_grad, cons_jac)
    elif scaling_type == ScalingType.KKT:
        lag_hess = problem.lag_hess(x0, y0)
        return Scaling.from_equilibrated_kkt(lag_hess, cons_jac)
    else:
        raise ValueError(f"Unknown scaling type {scaling_type}")
