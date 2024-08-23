from enum import Enum, auto
from typing import Callable

import numpy as np
import scipy as sp

from pygradflow.eval import EvalError


class BoxSolverError(Exception):
    pass


class BoxSolverStatus(Enum):
    Optimal = auto()
    Unbounded = auto()
    IterationLimit = auto()


class BoxSolverResult:
    def __init__(self, x, status, iterations):
        self.x = x
        self.status = status

    @property
    def success(self):
        return self.status == BoxSolverStatus.Optimal


# Based on "Projected Newton Methods for Optimization Problems with Simple Constraints"
def solve_box_constrained(
    x0: np.ndarray,
    func: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    hess: Callable[[np.ndarray], sp.sparse.spmatrix],
    lb: np.ndarray,
    ub: np.ndarray,
    obj_lower: float,
    max_it=1000,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> BoxSolverResult:

    (n,) = x0.shape
    assert lb.shape == (n,)
    assert ub.shape == (n,)

    curr_x = np.clip(x0, lb, ub)

    beta = 0.5
    sigma = 1e-3

    status = BoxSolverStatus.IterationLimit

    for iteration in range(max_it):
        curr_func = func(curr_x)
        curr_grad = grad(curr_x)

        if curr_func <= obj_lower:
            status = BoxSolverStatus.Unbounded
            break

        assert curr_grad.shape == (n,)

        at_lower = np.isclose(curr_x, lb)
        active_lower = np.logical_and(at_lower, curr_grad > 0)

        at_upper = np.isclose(curr_x, ub)
        active_upper = np.logical_and(at_upper, curr_grad < 0)

        residuals = -curr_grad
        residuals[at_lower] = np.maximum(residuals[at_lower], 0)
        residuals[at_upper] = np.minimum(residuals[at_upper], 0)

        residuum = np.linalg.norm(residuals, ord=np.inf)
        grad_norm = np.linalg.norm(curr_grad, ord=np.inf)

        if grad_norm < atol:
            status = BoxSolverStatus.Optimal
            break

        if (residuum < atol) or (residuum / grad_norm) < rtol:
            status = BoxSolverStatus.Optimal
            break

        active = np.logical_or(active_lower, active_upper)
        inactive = np.logical_not(active)

        dir = np.zeros((n,))
        inactive_grad = curr_grad[inactive]

        curr_hess = hess(curr_x)
        assert curr_hess.shape == (n, n)
        inactive_hess = curr_hess.tocsr()[inactive, :][:, inactive]
        dir[inactive] = sp.sparse.linalg.spsolve(inactive_hess, -inactive_grad)

        if np.dot(dir, curr_grad) >= 0:
            raise BoxSolverError("Inactive Hessian not positive definite")

        alpha = 1.0

        eval_errors = False

        for i in range(20):
            next_x = np.clip(curr_x + alpha * dir, lb, ub)

            try:
                next_func = func(next_x)
            except (ArithmeticError, EvalError):
                eval_errors = True
                alpha *= beta
                continue

            rhs = alpha * np.dot(curr_grad[inactive], dir[inactive])

            rhs += np.dot(curr_grad[active], curr_x[active] - next_x[active])

            func_diff = curr_func - next_func

            if func_diff >= sigma * rhs:
                break

            alpha *= beta
        else:
            if eval_errors:
                raise BoxSolverError("Line search failed")
            raise Exception("Line search did not converge")

        curr_x = next_x

    else:
        raise BoxSolverError(f"Did not converge after {max_it} iterations")

    return BoxSolverResult(curr_x, status, iteration)
