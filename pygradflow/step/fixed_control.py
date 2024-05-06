from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.step_control import StepController, StepControlResult


class FixedStepSizeController(StepController):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)
        self.lamb = params.lamb_init

    def step(self, iterate, rho, dt, next_steps, display, timer):
        assert dt > 0.0

        step = next(next_steps)

        return StepControlResult.from_step_result(step, self.lamb, True)
