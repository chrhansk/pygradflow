import numpy as np
import scipy as sp

from pygradflow.problem import Problem
from pygradflow.solver import Solver, SolverStatus


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
            return np.array([1.0] + [0.0] * (num_vars - 1))

        def cons(self, x):
            return np.array([])

        def cons_jac(self, x):
            return sp.sparse.coo_matrix((0, num_vars))

        def lag_hess(self, x, lag):
            return sp.sparse.diags([0.0] * num_vars)

    problem = UnboundedProblem()

    solver = Solver(problem)

    x0 = np.array([0.0] * num_vars)
    y0 = np.array([])

    result = solver.solve(x0, y0)

    assert result.status == SolverStatus.Unbounded


def test_cons_bounds_infeasible():
    num_vars = 1
    num_cons = 1

    class InfeasibleProblem(Problem):
        def __init__(self):
            var_lb = np.full(shape=(num_vars,), fill_value=0.0)
            var_ub = np.full(shape=(num_vars,), fill_value=100.0)

            super().__init__(
                var_lb, var_ub, cons_lb=np.array([1.0]), cons_ub=np.array([np.inf])
            )

        def obj(self, x):
            return 10.0

        def obj_grad(self, x):
            return np.zeros_like(x)

        def cons(self, x):
            return -x[0:1]

        def cons_jac(self, x):
            jac = np.zeros((num_cons, num_vars))
            jac[0, 0] = -1
            return sp.sparse.coo_matrix(jac)

        def lag_hess(self, x, y):
            hess = np.zeros((num_vars, num_vars))
            return sp.sparse.coo_matrix(hess)

    problem = InfeasibleProblem()

    solver = Solver(problem)

    x0 = np.array([10.0] * num_vars)
    y0 = np.array([0.0])

    result = solver.solve(x0, y0)

    assert np.array_equal(result.x, np.array([0.0]))

    assert result.status == SolverStatus.LocallyInfeasible


def test_detect_infeasible():
    num_vars = 1
    num_cons = 1

    class InfeasibleProblem(Problem):
        def __init__(self):
            var_lb = np.full(shape=(num_vars,), fill_value=-np.inf)
            var_ub = np.full(shape=(num_vars,), fill_value=np.inf)
            super().__init__(var_lb, var_ub, num_cons=num_cons)

        def obj(self, x):
            return x[0]

        def obj_grad(self, x):
            return np.array([1.0] + [0.0] * (num_vars - 1))

        def cons(self, x):
            x = x.item()
            return np.array([x * x + 1])

        def cons_jac(self, x):
            x = x.item()
            jac = np.array([[2 * x]])
            return sp.sparse.coo_matrix(jac)

        def lag_hess(self, x, lag):
            return sp.sparse.diags([2.0] * num_vars)

    problem = InfeasibleProblem()

    solver = Solver(problem)

    x0 = np.array([0.0] * num_vars)
    y0 = np.array([0.0])

    result = solver.solve(x0, y0)

    assert result.status == SolverStatus.LocallyInfeasible
