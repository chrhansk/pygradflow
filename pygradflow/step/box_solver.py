import numpy as np
import scipy as sp


# Simple dense BFGS implementation
# Note: We should use a sparse limited-memory variant
# storing the approximate inverse Hessian
class DampedBFGS:
    def __init__(self, n):
        self.mat = np.eye(n)

    def update(self, s, y):
        assert np.linalg.norm(s) > 0

        s_prod = np.dot(self.mat, s)

        prod = np.dot(s, y)
        bidir_prod = np.dot(s, s_prod)

        assert bidir_prod >= 0.0

        if prod >= 0.2 * bidir_prod:
            theta = 1
        else:
            theta = 0.8 * bidir_prod / (bidir_prod - prod)

        r = theta * y + (1 - theta) * s_prod

        assert np.dot(r, s) > 0

        self.mat -= np.outer(s_prod, s_prod) / bidir_prod
        self.mat += np.outer(r, r) / np.dot(r, s)


class BoxSolverError(Exception):
    pass


# Based on "Projected Newton Methods for Optimization Problems with Simple Constraints"
def solve_box_constrained(x0, func, grad, hess, lb, ub, max_it=1000, use_bfgs=False):

    (n,) = x0.shape
    assert lb.shape == (n,)
    assert ub.shape == (n,)

    curr_x = np.clip(x0, lb, ub)

    beta = 0.5
    sigma = 1e-3

    if use_bfgs:
        bfgs = DampedBFGS(n)

    prev_x = None
    prev_grad = None

    for iteration in range(max_it):
        curr_func = func(curr_x)
        curr_grad = grad(curr_x)

        if prev_x is not None and use_bfgs:
            s = curr_x - prev_x
            y = curr_grad - prev_grad

            bfgs.update(s, y)

        assert curr_grad.shape == (n,)

        at_lower = np.isclose(curr_x, lb)
        active_lower = np.logical_and(at_lower, curr_grad > 0)

        at_upper = np.isclose(curr_x, ub)
        active_upper = np.logical_and(at_upper, curr_grad < 0)

        residuum = -curr_grad
        residuum[at_lower] = np.maximum(residuum[at_lower], 0)
        residuum[at_upper] = np.minimum(residuum[at_upper], 0)

        if np.linalg.norm(residuum, ord=np.inf) < 1e-8:
            print(f"Converged after {iteration} iterations")
            break

        active = np.logical_or(active_lower, active_upper)
        inactive = np.logical_not(active)

        dir = np.zeros((n,))
        inactive_grad = curr_grad[inactive]

        if use_bfgs:
            curr_hess = bfgs.mat
            inactive_grad = curr_grad[inactive]
            inactive_hess = curr_hess[inactive, :][:, inactive]
            dir[inactive] = np.linalg.solve(inactive_hess, -inactive_grad)
        else:
            curr_hess = hess(curr_x)
            assert curr_hess.shape == (n, n)
            inactive_hess = curr_hess.tocsr()[inactive, :][:, inactive]
            dir[inactive] = sp.sparse.linalg.spsolve(inactive_hess, -inactive_grad)

        if np.dot(dir, curr_grad) >= 0:
            raise BoxSolverError("Hessian not positive definite")

        alpha = 1.0

        for i in range(20):
            next_x = np.clip(curr_x + alpha * dir, lb, ub)
            next_func = func(next_x)

            rhs = alpha * np.dot(curr_grad[inactive], dir[inactive])

            rhs += np.dot(curr_grad[active], curr_x[active] - next_x[active])

            func_diff = curr_func - next_func

            if func_diff >= sigma * rhs:
                break

            alpha *= beta
        else:
            raise Exception("Line search failed")

        prev_grad = curr_grad
        prev_x = curr_x

        curr_x = next_x

    else:
        raise BoxSolverError(f"Did not converge after {max_it} iterations")

    return curr_x
