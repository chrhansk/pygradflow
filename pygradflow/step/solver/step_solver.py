import abc
import functools
from typing import Optional, cast

import numpy as np
import scipy as sp

from pygradflow.implicit_func import StepFunc
from pygradflow.iterate import Iterate
from pygradflow.linear_solver import LinearSolver, LinearSolverError
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.util import norm_mult


class StepResult:
    def __init__(self, orig_iterate, dx, dy, active_set, rcond=None):
        self.orig_iterate = orig_iterate
        self.dy = dy
        self.active_set = active_set
        self.rcond = rcond

        self._compute_xn(dx)

    def _compute_xn(self, dx):
        iterate = self.orig_iterate
        problem = iterate.problem

        var_lb = problem.var_lb
        var_ub = problem.var_ub

        xn = iterate.x - dx
        dx = np.copy(dx)

        at_lb = xn < var_lb
        xn[at_lb] = var_lb[at_lb]
        dx[at_lb] = iterate.x[at_lb] - var_lb[at_lb]

        at_ub = xn > var_ub
        xn[at_ub] = var_ub[at_ub]
        dx[at_ub] = iterate.x[at_ub] - var_ub[at_ub]

        self.dx = dx
        self.xn = xn

    @functools.cached_property
    def iterate(self):
        iterate = self.orig_iterate

        xn = self.xn.astype(iterate.x.dtype, copy=False)
        yn = iterate.y - self.dy

        return Iterate(
            iterate.problem,
            iterate.params,
            xn,
            yn,
            iterate.eval,
        )

    @functools.cached_property
    def diff(self):
        return norm_mult(self.dx, self.dy)


class StepSolver(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params
        self.n = problem.num_vars
        self.m = problem.num_cons

        self._active_set: Optional[np.ndarray] = None
        self._jac: Optional[sp.sparse.spmatrix] = None
        self._hess: Optional[sp.sparse.spmatrix] = None

        self.solver: Optional[LinearSolver] = None

    @property
    def active_set(self) -> np.ndarray:
        assert self._active_set is not None
        return cast(np.ndarray, self._active_set)

    @property
    def jac(self) -> sp.sparse.spmatrix:
        assert self._jac is not None
        return cast(sp.sparse.spmatrix, self._jac)

    @property
    def hess(self) -> sp.sparse.spmatrix:
        assert self._hess is not None
        return cast(sp.sparse.spmatrix, self._hess)

    def linear_solver(self, mat: sp.sparse.spmatrix) -> LinearSolver:
        from pygradflow.linear_solver import linear_solver

        solver_type = self.params.linear_solver_type
        return linear_solver(mat, solver_type)

    def estimate_rcond(
        self, mat: sp.sparse.spmatrix, solver: LinearSolver
    ) -> Optional[float]:
        from pygradflow.step.cond_estimate import ConditionEstimator

        estimator = ConditionEstimator(mat, solver, self.params)
        rcond = None

        try:
            rcond = estimator.estimate_rcond()
        except LinearSolverError:
            pass

        return rcond

    @abc.abstractmethod
    def update_active_set(self, active_set: np.ndarray):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def func(self) -> StepFunc:
        raise NotImplementedError()

    @abc.abstractmethod
    def update_derivs(self, iterate: Iterate):
        raise NotImplementedError()

    @abc.abstractmethod
    def solve(self, iterate: Iterate) -> StepResult:
        raise NotImplementedError()
