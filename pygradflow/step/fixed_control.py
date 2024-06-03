from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.newton_control import NewtonController
from pygradflow.step.step_control import StepControlResult


class FixedStepSizeController(NewtonController):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)
        self.lamb = params.lamb_init

    def step(self, iterate, rho, dt, display, timer):
        assert dt > 0.0

        next_steps = self.newton_steps(iterate, rho, dt)

        step = next(next_steps)

        return StepControlResult.from_step_result(step, self.lamb, True)
