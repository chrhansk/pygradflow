import math

import numpy as np
import scipy as sp

from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.step.linear_solver import LinearSolver

seed = 42


class ConditionEstimator:
    """
    Estimates condition of matrix using generic lienar solver.
    Based on
    "Estimating Extremal Eigenvalues and Condition Numbers of Matrices"
    by Dixon
    """

    def __init__(
        self,
        mat: sp.sparse.spmatrix,
        linear_solver: LinearSolver,
        params: Params,
        min_prob: float = 0.99,
        factor: float = 10.0,
    ):
        assert 0 < min_prob < 1
        self.size = mat.shape[0]

        self.params = params
        assert mat.shape == (self.size, self.size)

        self.min_prob = min_prob
        self.factor = factor
        self.mat = mat
        self.linear_solver = linear_solver
        self.rng = np.random.default_rng(seed=seed)

    def _required_its(self):
        factor = (1.0 - self.min_prob) / 1.6 * math.pow(self.size, -0.5)
        return -2 * math.ceil(math.log(factor, self.factor))

    def _random_vec(self):
        rng = self.rng
        size = self.size
        vec = rng.normal(size=size)

        while True:
            if (vec != 0.0).any():
                break
            vec = rng.normal(size=size)

        norm = np.linalg.norm(vec)
        return (vec / norm).astype(self.params.dtype)

    def estimate_rcond(self) -> float:
        mat = self.mat
        trans_mat = mat.T
        linear_solver = self.linear_solver

        num_its = self._required_its()
        assert num_its > 0

        x = self._random_vec()
        y = self._random_vec()

        xprod = np.copy(x)
        yprod = np.copy(y)

        # Factors to prevent over- or underflow
        xfac = 1.0
        yfac = 1.0

        logger.debug("Number of iterations: %s", num_its)

        for k in range(num_its):
            xprod = mat @ xprod
            xprod = trans_mat @ xprod

            yprod = linear_solver.solve(yprod, trans=True)
            yprod = linear_solver.solve(yprod)

            xnorm = np.linalg.norm(xprod)
            ynorm = np.linalg.norm(yprod)

            xfac *= xnorm
            xprod /= xnorm

            yfac *= ynorm
            yprod /= ynorm

            assert y.dot(yprod) > 0.0

        pow_fac = 1.0 / (2.0 * num_its)

        xdot = x.dot(xprod) * xfac
        xdot = math.pow(xdot, pow_fac)

        ydot = y.dot(yprod) * yfac
        ydot = math.pow(ydot, pow_fac)

        if np.isinf(xdot) or np.isinf(ydot):
            return 0.0

        cond = xdot * ydot

        if np.isinf(cond):
            return 0.0

        rcond = 1.0 / cond

        return rcond
