import functools
from typing import Optional

import numpy as np

from pygradflow.cons_problem import ConstrainedProblem
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.scale import ScaledProblem, Scaling


class Transformation:
    def __init__(
        self, orig_problem: Problem, params: Params, scaling: Optional[Scaling] = None
    ):
        self.orig_problem = orig_problem
        self.params = params
        self.scaling = scaling

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
