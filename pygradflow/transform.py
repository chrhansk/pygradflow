import functools
from typing import Tuple

import numpy as np

from pygradflow.cons_problem import ConstrainedProblem
from pygradflow.eval import create_evaluator
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.scale import ScaledProblem, create_scaling


class Transformation:
    def __init__(
        self,
        orig_problem: Problem,
        params: Params,
        x0: np.ndarray | float | None,
        y0: np.ndarray | float | None,
    ):
        self.orig_problem = orig_problem

        (self.x0, self.y0) = self._create_initial_values(orig_problem, x0, y0)
        self.scaling = create_scaling(orig_problem, params, self.x0, self.y0)
        self.params = params

        self.evaluator = create_evaluator(self.trans_problem, params)

    def _create_initial_values(
        self,
        orig_problem,
        x0: np.ndarray | float | None,
        y0: np.ndarray | float | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_lb = orig_problem.var_lb
        orig_ub = orig_problem.var_ub

        if x0 is None:
            x0 = np.clip(0.0, orig_problem.var_lb, orig_problem.var_ub)
        else:
            if (x0 > orig_ub).any() or (x0 < orig_lb).any():
                logger.warning("Initial point violates variable bounds")
                x0 = np.clip(x0, orig_lb, orig_ub)

        if y0 is None:
            y0 = 0.0

        x0 = np.broadcast_to(x0, shape=(orig_problem.num_vars,))
        y0 = np.broadcast_to(y0, shape=(orig_problem.num_cons,))

        return (x0, y0)

    @functools.cached_property
    def scaled_problem(self):
        orig_problem = self.orig_problem
        scaling = self.scaling

        if scaling is None:
            return orig_problem

        return ScaledProblem(orig_problem, scaling)

    @functools.cached_property
    def trans_problem(self):
        scaled_problem = self.scaled_problem

        return ConstrainedProblem(scaled_problem)

    @property
    def initial_iterate(self):
        (x0, y0) = self.transform_sol(self.x0, self.y0)
        dtype = self.params.dtype

        x = x0.astype(dtype)
        y = y0.astype(dtype)

        return Iterate(self.trans_problem, self.params, x, y, self.evaluator)

    def transform_sol(self, x: np.ndarray, y: np.ndarray):
        scaling = self.scaling

        if scaling is None:
            scaled_x = x
            scaled_y = y
        else:
            scaled_x = scaling.scale_primal(x)
            scaled_y = scaling.scale_dual(y)

        trans_problem = self.trans_problem

        return trans_problem.transform_sol(scaled_x, scaled_y)

    def restore_sol(self, x: np.ndarray, y: np.ndarray, d: np.ndarray):
        trans_problem = self.trans_problem

        (x, y, d) = trans_problem.restore_sol(x, y, d)

        scaling = self.scaling

        if scaling is None:
            return (x, y, d)

        orig_x = scaling.unscale_primal(x)
        orig_y = scaling.unscale_dual(y)
        orig_d = scaling.unscale_bounds_dual(d)

        return (orig_x, orig_y, orig_d)
