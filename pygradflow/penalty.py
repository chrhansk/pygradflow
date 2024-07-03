import abc
from typing import List, Tuple

import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, PenaltyUpdate
from pygradflow.problem import Problem


class PenaltyResult:
    def __init__(self, next_rho, accept) -> None:
        self.next_rho = next_rho
        self.accept = accept

    @staticmethod
    def accept_with_penalty(next_rho):
        return PenaltyResult(next_rho, True)

    @staticmethod
    def reject_with_penalty(next_rho):
        return PenaltyResult(next_rho, False)


class PenaltyStrategy(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params

    def initial(self, iterate) -> float:
        return self.params.rho

    def update(self, prev_iterate, next_iterate) -> PenaltyResult:
        raise NotImplementedError()


class ConstantPenalty(PenaltyStrategy):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> PenaltyResult:
        return PenaltyResult.accept_with_penalty(self.params.rho)


class DualNormUpdate(PenaltyStrategy):
    """
    Increase penalty such that it is within a factor of the inf-norm
    of the dual variables
    """

    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def initial(self, iterate: Iterate) -> float:
        self.rho = self.params.rho
        return self.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> PenaltyResult:
        iterate = next_iterate

        if self.problem.num_cons == 0:
            return PenaltyResult.accept_with_penalty(self.rho)

        ynorm = float(np.linalg.norm(iterate.y, ord=np.inf))

        assert ynorm >= 0.0

        if ynorm >= 10.0 * self.rho:
            next_rho = min(ynorm, 10.0 * self.rho)
            assert next_rho > self.rho
            self.rho = next_rho

        return PenaltyResult.accept_with_penalty(self.rho)


class DualEquilibration(PenaltyStrategy):
    """
    Increase penalty such that the dual the violation is within
    a factor of the product of the dual variables and the constraint values
    """

    def __init__(self, problem: Problem, params: Params):
        super().__init__(problem, params)

    def initial(self, iterate: Iterate) -> float:
        self.rho = self.params.rho
        return self.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> PenaltyResult:
        iterate = next_iterate

        cons = iterate.cons

        yprod = abs(np.dot(iterate.y, cons))
        viol = 1.0 / 2.0 * np.dot(cons, cons)

        assert yprod >= 0.0
        assert viol >= 0.0

        if viol == 0.0:
            return PenaltyResult.accept_with_penalty(self.rho)

        target_rho = 0.01 * yprod / viol

        if self.rho < target_rho:
            next_rho = max(self.rho * 10.0, target_rho)
            assert next_rho > self.rho

            self.rho = next_rho

        return PenaltyResult.accept_with_penalty(self.rho)


class ParetoDecrease(PenaltyStrategy):
    """
    Increase penalty such that at least one of objective
    and constraint violation is (weakly) reduced in the direction
    of the curve
    """

    def __init__(self, problem: Problem, params: Params):
        super().__init__(problem, params)

        if problem.var_bounded:
            logger.warning(
                "Pareto decrease penalty update may not work with variable bounds"
            )

    def initial(self, iterate: Iterate) -> float:
        self.rho = self.params.rho
        return self.rho

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> PenaltyResult:
        iterate = next_iterate
        params = self.params

        cons = iterate.cons

        viol = 1.0 / 2.0 * np.dot(cons, cons)

        # Don't update if we are already feasible
        if viol <= params.opt_tol:
            return PenaltyResult.accept_with_penalty(self.rho)

        cons_jac = iterate.cons_jac

        infeas_opt_res = cons_jac.T.dot(cons)

        # Cannot find bound if we are locally infeasible
        if np.linalg.norm(infeas_opt_res, ord=np.inf) <= params.local_infeas_tol:
            return PenaltyResult.accept_with_penalty(self.rho)

        obj_bound = np.inf

        obj_grad = iterate.obj_grad
        obj_prod = np.dot(obj_grad, infeas_opt_res)

        cons_dual_prod = cons_jac.T.dot(iterate.y)

        if abs(obj_prod) > 1e-10:
            obj_grad_norm = np.linalg.norm(obj_grad)
            lhs = -(obj_grad_norm + cons_dual_prod.dot(obj_grad))
            obj_bound = lhs / obj_prod

        infeas_res_norm = np.linalg.norm(infeas_opt_res)

        lhs = -np.dot(infeas_opt_res, obj_grad + cons_dual_prod)

        cons_bound = lhs / infeas_res_norm

        bound = min(obj_bound, cons_bound)

        assert np.isfinite(bound)

        next_rho = min(self.rho * 10.0, bound)
        next_rho = max(next_rho, self.rho)

        assert next_rho >= self.rho

        self.rho = next_rho

        return PenaltyResult.accept_with_penalty(self.rho)


class PenaltyFilter(PenaltyStrategy):
    """
    Filter maintaining a set of Pareto optimal points of
    objective and constraint violation. The penalty is
    increased at a point if it is dominated by any
    other point in the filter.

    Note: The running time is linear in the size of the
    filter, this could be improved to be logarithmic
    """

    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)
        self.entries: List[Tuple[float, float]] = []
        self.rho = self.params.rho

    def filter_insert(self, obj, violation) -> bool:
        entry = (obj, violation)

        def dominates(first, second):
            return first[0] <= second[0] and first[1] <= second[1]

        if any(dominates(e, entry) for e in self.entries):
            return False

        self.entries = [e for e in self.entries if not dominates(entry, e)]
        self.entries.append(entry)

        return True

    def update(self, prev_iterate, next_iterate) -> PenaltyResult:
        next_obj = next_iterate.obj
        next_violation = next_iterate.cons_violation

        if self.filter_insert(next_obj, next_violation):
            return PenaltyResult.accept_with_penalty(self.rho)

        return PenaltyResult.reject_with_penalty(10.0 * self.rho)


def penalty_strategy(problem: Problem, params: Params) -> PenaltyStrategy:
    penalty_update = params.penalty_update

    if penalty_update == PenaltyUpdate.Constant:
        return ConstantPenalty(problem, params)
    elif penalty_update == PenaltyUpdate.DualNorm:
        return DualNormUpdate(problem, params)
    elif penalty_update == PenaltyUpdate.DualEquilibration:
        return DualEquilibration(problem, params)
    elif penalty_update == PenaltyUpdate.ParetoDecrease:
        return ParetoDecrease(problem, params)
    elif penalty_update == PenaltyUpdate.Filter:
        return PenaltyFilter(problem, params)

    raise ValueError("Invalid penalty update strategy")
