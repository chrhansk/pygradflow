import abc
import logging
from typing import Optional

import numpy as np

from pygradflow.display import StateData, inner_display
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, StepControlType
from pygradflow.problem import Problem
from pygradflow.step.solver.step_solver import StepResult
from pygradflow.step.step_solver_error import StepSolverError
from pygradflow.timer import Timer


class StepControlResult:
    def __init__(
        self,
        iterate: Iterate,
        lamb: float,
        active_set,
        rcond: Optional[float],
        accepted: bool,
    ) -> None:
        self.iterate = iterate
        self.lamb = lamb
        self.active_set = active_set
        self.rcond = rcond
        self.accepted = accepted

    @staticmethod
    def from_step_result(
        step_result: StepResult, lamb: float, accepted: bool
    ) -> "StepControlResult":
        return StepControlResult(
            step_result.iterate,
            lamb,
            step_result.active_set,
            step_result.rcond,
            accepted,
        )


class StepController(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params
        self.lamb = params.lamb_init

    @abc.abstractmethod
    def step(
        self,
        iterate: Iterate,
        rho: float,
        dt: float,
        display: bool,
        timer: Timer,
    ) -> StepControlResult:
        raise NotImplementedError()

    def update_stepsize_after_fail(self, lamb: float) -> float:
        return 2.0 * lamb

    def compute_step(
        self,
        iterate: Iterate,
        rho: float,
        dt: float,
        display: bool,
        timer: Timer,
    ) -> StepControlResult:
        """
        Computes next step using the step method, handling
        linear algebra errors by reducing the step size.
        """

        def fail_result():
            lamb = 1.0 / dt
            lamb = self.update_stepsize_after_fail(lamb)
            return StepControlResult(iterate, lamb, None, None, False)

        try:
            self.display = None
            if display:

                def res_func(next_iterate):
                    func = ImplicitFunc(self.problem, iterate, dt)
                    return np.linalg.norm(func.value_at(next_iterate, rho))

                self.res_func = res_func
                self.display = inner_display(self.problem, self.params)
                logger.debug("     %s", self.display.header)
            return self.step(iterate, rho, dt, display, timer)
        except StepSolverError as e:
            logger.debug("Step solver error during step computation: %s", e)
            return fail_result()

    def display_step(self, iteration, step):
        level = logger.getEffectiveLevel()
        if not self.display or level > logging.DEBUG:
            return

        iterate = step.iterate
        state = StateData()
        state["iter"] = lambda: iteration
        state["residuum"] = lambda: self.res_func(iterate)
        state["dist"] = lambda: step.diff
        state["active_set_size"] = lambda: step.active_set.sum()
        logger.debug("     %s", self.display.row(state))


def step_controller(problem: Problem, params: Params) -> StepController:
    step_control_type = params.step_control_type

    if step_control_type == StepControlType.Exact:
        from pygradflow.step.exact_control import ExactController

        return ExactController(problem, params)
    elif step_control_type == StepControlType.Fixed:
        from pygradflow.step.fixed_control import FixedStepSizeController

        return FixedStepSizeController(problem, params)
    elif step_control_type == StepControlType.ResiduumRatio:
        from pygradflow.step.residuum_ratio_control import ResiduumRatioController

        return ResiduumRatioController(problem, params)
    elif step_control_type == StepControlType.Optimizing:
        from pygradflow.step.opti_control import OptimizingController

        return OptimizingController(problem, params)
    elif step_control_type == StepControlType.BoxReduced:
        from pygradflow.step.box_control import BoxReducedController

        return BoxReducedController(problem, params)
    else:
        assert step_control_type == StepControlType.DistanceRatio
        from pygradflow.step.distance_ratio_control import DistanceRatioController

        return DistanceRatioController(problem, params)
