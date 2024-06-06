import pytest

import scipy as sp
import numpy as np


from pygradflow.params import LinearSolverType
from pygradflow.linear_solver import LinearSolverError, linear_solver


symm_indef_solvers = [
    LinearSolverType.LU,
    LinearSolverType.MINRES,
    LinearSolverType.GMRES,
    # LinearSolverType.Cholesky,
    LinearSolverType.MA57,
    LinearSolverType.SSIDS,
    LinearSolverType.MUMPS,
]


def _indef_mat():
    mat = np.array([[ 2,  1,  0,  0,  0 ],
                    [ 1,  4,  1,  0,  1, ],
                    [ 0,  1,  3,  2,  0 ],
                    [ 0,  0,  2, -1,  0 ],
                    [ 0,  1,  0,  0,  2 ],],
                   dtype=float)

    return sp.sparse.csc_matrix(mat)


@pytest.fixture
def indef_matrix():
    return _indef_mat()


@pytest.fixture
def posdef_matrix():
    mat = _indef_mat()
    eigvals = np.linalg.eigvalsh(mat.toarray())
    min_eigval = eigvals.min()

    delta = min(2. * min_eigval, 0.)

    (_, n) = mat.shape
    posdef_mat = mat + sp.sparse.diags([-delta]*n, )

    return posdef_mat


@pytest.fixture
def negdef_matrix():
    mat = _indef_mat()
    eigvals = np.linalg.eigvalsh(mat.toarray())
    max_eigval = eigvals.max()

    delta = max(2. * max_eigval, 0.)

    (_, n) = mat.shape
    negdef_mat = mat + sp.sparse.diags([-delta]*n, )

    return negdef_mat


@pytest.fixture
def rhs():
    return np.array([4.0, 17.0, 19.0, 2.0, 12.0])


def compute_num_neg_eigvals(mat):
    eigvals = np.linalg.eigvalsh(mat.toarray())
    return (eigvals < 0).sum()


def get_linear_solver(matrix, linear_solver_type, symmetric):
    try:
        return linear_solver(matrix,
                             linear_solver_type,
                             symmetric=symmetric)
    except ImportError:
        pytest.skip(f"{linear_solver_type} not available")


@pytest.mark.parametrize("linear_solver_type", list(LinearSolverType))
def test_solve_symmetric_posdef(posdef_matrix, rhs, linear_solver_type):
    matrix = posdef_matrix

    solver = get_linear_solver(matrix, linear_solver_type, symmetric=True)
    sol = solver.solve(rhs)

    res = matrix.dot(sol) - rhs
    assert np.allclose(res, 0.0)

    num_neg_eigvals = solver.num_neg_eigvals()

    assert num_neg_eigvals in [compute_num_neg_eigvals(matrix), None]


@pytest.mark.parametrize("linear_solver_type", symm_indef_solvers)
def test_solve_symmetric_indef(indef_matrix, rhs, linear_solver_type):
    matrix = indef_matrix

    solver = get_linear_solver(matrix, linear_solver_type, symmetric=True)
    sol = solver.solve(rhs)

    res = matrix.dot(sol) - rhs
    assert np.allclose(res, 0.0)

    num_neg_eigvals = solver.num_neg_eigvals()

    assert num_neg_eigvals in [compute_num_neg_eigvals(matrix), None]


@pytest.mark.parametrize("linear_solver_type", symm_indef_solvers)
def test_solve_symmetric_negdef(negdef_matrix, rhs, linear_solver_type):
    matrix = negdef_matrix

    solver = get_linear_solver(matrix, linear_solver_type, symmetric=True)
    sol = solver.solve(rhs)

    res = matrix.dot(sol) - rhs
    assert np.allclose(res, 0.0)

    num_neg_eigvals = solver.num_neg_eigvals()

    assert num_neg_eigvals in [compute_num_neg_eigvals(matrix), None]
