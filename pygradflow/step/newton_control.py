from typing import Iterator, Optional

from pygradflow.iterate import Iterate
from pygradflow.newton import newton_method
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.step.solver.step_solver import StepResult
from pygradflow.step.step_control import StepController


class NewtonController(StepController):
    """
    Step controller working by solving the implicit Euler
    equations using a semi-smooth Newton method
    """

    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def newton_steps(
        self,
        iterate: Iterate,
        rho: float,
        dt: float,
        initial_iterate: Optional[Iterate] = None,
    ) -> Iterator[StepResult]:
        problem = self.problem
        params = self.params

        self.method = newton_method(problem, params, iterate, dt, rho)

        if initial_iterate is not None:
            curr_iterate = initial_iterate
        else:
            curr_iterate = iterate

        while True:
            next_step = self.method.step(curr_iterate)
            yield next_step
            curr_iterate = next_step.iterate
