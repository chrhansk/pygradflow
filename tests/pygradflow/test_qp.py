import numpy as np
import pytest
import scipy as sp

from pygradflow.params import NewtonType, Params, StepControlType
from pygradflow.solver import Solver
from pygradflow.status import SolverStatus

from .qp import QP


@pytest.fixture
def unbounded_qp():
    n = 199
    h = 1 / n
    e = np.ones(n)
    H = (1 / h**2) * sp.sparse.spdiags([e, -2 * e, e], [-1, 0, 1], n, n, format="coo")
    A = sp.sparse.coo_matrix((0, n))
    g = -e
    b = np.array([])
    lb = -np.inf * e
    lb[n // 4] = 0.0
    lb[3 * n // 4] = 0.0
    lb[n // 2] = 0.0
    ub = np.inf * e
    return QP(H, sp.sparse.coo_matrix((0, n)), g, np.array([]), lb=lb, ub=ub)


@pytest.fixture
def boxed_qp():
    n = 49
    h = 1 / n
    e = np.ones(n)
    H = (1 / h**2) * sp.sparse.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format="coo")
    g = e

    lb = np.linspace(0, -0.01, n + 2)[1:-1]
    lb[n // 4] = 0.0
    lb[3 * n // 4] = 0.0
    lb[n // 2] = 0.0
    ub = np.inf * e
    return QP(H, sp.sparse.coo_matrix((0, n)), g, np.array([]), lb=lb, ub=ub)


@pytest.fixture
def unbounded_x0():
    return 0.0


@pytest.mark.parametrize(
    "step_control_type",
    [
        StepControlType.Exact,
        StepControlType.Optimizing,
        StepControlType.BoxReduced,
        StepControlType.ResiduumRatio,
        # StepControlType.DistanceRatio,
    ],
)
def test_unbounded(unbounded_qp, unbounded_x0, step_control_type):

    problem = unbounded_qp

    params = Params(
        step_control_type=step_control_type, iteration_limit=1000, display_interval=0.0
    )

    solver = Solver(problem, params)

    result = solver.solve(x0=np.amax(problem.var_lb, 0), y0=np.array([]))

    assert result.status == SolverStatus.Unbounded


def test_boxed(boxed_qp, unbounded_x0):
    problem = boxed_qp

    params = Params(
        iteration_limit=1000,
        display_interval=0.0,
        # newton_type=NewtonType.Full,
        # lamb_init=1e-12,
    )

    solver = Solver(problem, params)

    result = solver.solve(x0=0.0, y0=np.array([]))

    assert result.status == SolverStatus.Optimal


@pytest.mark.parametrize(
    "newton_type",
    [
        NewtonType.ActiveSet,
        NewtonType.Full,
        # NewtonType.Globalized,
        NewtonType.Simplified,
    ],
)
def test_newton_types(newton_type):
    n = 49
    h = 1 / n
    e = np.ones(n)
    H = (1 / h**2) * sp.sparse.spdiags([-e, 2 * e, -e], [-1, 0, 1], n, n, format="coo")
    A = sp.sparse.coo_matrix((0, n))
    g = e
    b = np.array([])
    lb = -np.inf * e
    ub = np.inf * e
    qp = QP(H, A, g, b, lb, ub)

    lb = np.linspace(0, -0.01, n + 2)[1:-1]
    lb[n // 4] = 0.0
    lb[3 * n // 4] = 0.0
    lb[n // 2] = 0.0
    ub = np.inf * e
    qp = QP(H, sp.sparse.coo_matrix((0, n)), g, np.array([]), lb=lb, ub=ub)

    params = Params()
    params.display_interval = 1e-16
    params.lamb_init = 1e-12
    params.iteration_limit = 1000
    params.newton_type = newton_type

    solver = Solver(problem=qp, params=params)
    result = solver.solve(x0=np.amax(lb, 0), y0=np.array([]))

    assert result.success
