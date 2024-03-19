import numpy as np

from pygradflow.problem import Problem


class Scaling:
    def __init__(self, var_weights, cons_weights, obj_weight=1):
        self.var_weights = var_weights
        self.cons_weights = cons_weights

        assert var_weights.ndim == 1
        assert var_weights.dtype == int

        assert cons_weights.ndim == 1
        assert cons_weights.dtype == int

        self.obj_weight = obj_weight

    @staticmethod
    def from_nominal_values(var_values, cons_values, obj_value):
        var_weights = Scaling.weights_from_nominal_values(var_values)
        cons_weights = Scaling.weights_from_nominal_values(cons_values)
        obj_weight = Scaling.weights_from_nominal_values(obj_value)

        return Scaling(var_weights, cons_weights, obj_weight)

    @staticmethod
    def weights_from_nominal_values(values):
        return 1 - np.frexp(values)[1]

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
        var_weights = self.scaling.var_weights
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
        var_weights = self.scaling.var_weights
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
