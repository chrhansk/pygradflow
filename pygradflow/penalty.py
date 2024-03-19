import abc

import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.params import Params, PenaltyUpdate
from pygradflow.problem import Problem


class PenaltyStrategy(abc.ABC):
    def __init__(self, problem: Problem, params: Params) -> None:
        self.problem = problem
        self.params = params

    def initial(self, iterate) -> float:
        return self.params.rho

    def update(self, prev_iterate, next_iterate) -> float:
        raise NotImplementedError()


class ConstantPenalty(PenaltyStrategy):
    def __init__(self, problem: Problem, params: Params) -> None:
        super().__init__(problem, params)

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        return self.params.rho


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

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        iterate = next_iterate

        if self.problem.num_cons == 0:
            return self.rho

        ynorm = float(np.linalg.norm(iterate.y, ord=np.inf))

        assert ynorm >= 0.0

        if ynorm >= 10.0 * self.rho:
            next_rho = min(ynorm, 10.0 * self.rho)
            assert next_rho > self.rho
            self.rho = next_rho

        return self.rho


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

    def update(self, prev_iterate: Iterate, next_iterate: Iterate) -> float:
        iterate = next_iterate
        params = self.params

        cons = iterate.cons

        viol = 1.0 / 2.0 * np.dot(cons, cons)

        # Don't update if we are already feasible
        if viol <= params.opt_tol:
            return self.rho

        cons_jac = iterate.cons_jac

        infeas_opt_res = cons_jac.T.dot(cons)

        # Cannot find bound if we are locally infeasible
        if np.linalg.norm(infeas_opt_res, ord=np.inf) <= params.local_infeas_tol:
            return self.rho

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
        return self.rho


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

    raise ValueError("Invalid penalty update strategy")
