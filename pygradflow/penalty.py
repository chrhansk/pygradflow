import abc

import numpy as np
from pygradflow.iterate import Iterate
from pygradflow.params import Params, PenaltyUpdate
from pygradflow.problem import Problem


class PenaltyStrategy(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params

    def initial(self, iterate) -> float:
        raise NotImplementedError()

    def update(self, prev_iterate, next_iterate) -> float:
        raise NotImplementedError()


class ConstantPenalty(PenaltyStrategy):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def initial(self, iterate: Iterate) -> float:
        return self.params.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        return self.params.rho


class DualNormUpdate(PenaltyStrategy):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def initial(self, iterate: Iterate) -> float:
        self.rho = self.params.rho
        return self.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        iterate = next_iterate

        if self.problem.num_cons == 0:
            return self.rho

        ynorm = np.linalg.norm(iterate.y, ord=np.inf)

        assert ynorm >= 0.0

        if ynorm >= 10.0 * self.rho:
            next_rho = min(ynorm, 10.0 * self.rho)
            assert next_rho > self.rho
            self.rho = next_rho

        return self.rho


class DualEquilibration(PenaltyStrategy):
    def __init__(self, problem: Problem, params: Params):
        super().__init__(problem, params)

    def initial(self, iterate: Iterate) -> float:
        self.rho = self.params.rho
        return self.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        iterate = next_iterate

        cons = iterate.cons

        yprod = abs(np.dot(iterate.y, cons))
        viol = 1.0 / 2.0 * np.dot(cons, cons)

        assert yprod >= 0.0
        assert viol >= 0.0

        if viol == 0.0:
            return self.rho

        target_rho = 0.01 * yprod / viol

        if self.rho < target_rho:
            next_rho = max(self.rho * 10.0, target_rho)
            assert next_rho > self.rho

            self.rho = next_rho

        return self.rho


def penalty_strategy(problem: Problem, params: Params) -> PenaltyStrategy:
    penalty_update = params.penalty_update

    if penalty_update == PenaltyUpdate.Constant:
        return ConstantPenalty(problem, params)
    elif penalty_update == PenaltyUpdate.DualNorm:
        return DualNormUpdate(problem, params)
    elif penalty_update == PenaltyUpdate.DualEquilibration:
        return DualEquilibration(problem, params)

    raise ValueError("Invalid penalty update strategy")
