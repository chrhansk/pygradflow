import abc
import math
from enum import Enum, auto

import numpy as np
import scipy as sp

from pygradflow.log import logger


def astype(array, dtype):
    if array.dtype == dtype:
        return array
    else:
        return array.astype(dtype)


class EvalError(ValueError):
    def __init__(self, msg, x):
        self.x = x
        super().__init__(msg)


def warn_once(*args):
    has_warned = [False]

    def warn():
        if not has_warned[0]:
            logger.warning(*args)
            has_warned[0] = True

    return warn


warn_hessian_pattern = warn_once("Unsymmetric Hessian pattern")
warn_hessian_values = warn_once("Hessian not numerically symmetric")


class Component(Enum):
    Obj = auto()
    ObjGrad = auto()
    Cons = auto()
    ConsJac = auto()
    LagHess = auto()

    def name(self):
        return {
            Component.Obj: "Objective",
            Component.ObjGrad: "Objective Gradient",
            Component.Cons: "Constraints",
            Component.ConsJac: "Constraint Jacobian",
            Component.LagHess: "Lagrangian Hessian",
        }[self]


class Evaluator(abc.ABC):
    def __init__(self, problem, params):
        self.problem = problem
        self.dtype = params.dtype

        self.reset_num_evals()

    def reset_num_evals(self):
        self.num_evals = {comp: 0 for comp in Component}

    def obj(self, x: np.ndarray) -> float:
        self.num_evals[Component.Obj] += 1
        return self._eval_obj(x)

    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        self.num_evals[Component.ObjGrad] += 1
        return self._eval_obj_grad(x)

    def cons(self, x: np.ndarray) -> np.ndarray:
        self.num_evals[Component.Cons] += 1
        return self._eval_cons(x)

    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        self.num_evals[Component.ConsJac] += 1
        return self._eval_cons_jac(x)

    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        self.num_evals[Component.LagHess] += 1
        return self._eval_lag_hess(x, lag)

    @abc.abstractmethod
    def _eval_obj(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def _eval_obj_grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _eval_cons(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _eval_cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()

    @abc.abstractmethod
    def _eval_lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()


class SimpleEvaluator(Evaluator):
    def _eval_obj(self, x: np.ndarray) -> float:
        return self.problem.obj(x)

    def _eval_obj_grad(self, x: np.ndarray) -> np.ndarray:
        return astype(self.problem.obj_grad(x), self.dtype)

    def _eval_cons(self, x: np.ndarray) -> np.ndarray:
        if self.problem.num_cons == 0:
            return np.array([], dtype=self.dtype)

        return astype(self.problem.cons(x), self.dtype)

    def _eval_cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        if self.problem.num_cons == 0:
            return sp.sparse.csr_matrix((0, self.problem.num_vars), dtype=self.dtype)

        return astype(self.problem.cons_jac(x), self.dtype)

    def _eval_lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        return astype(self.problem.lag_hess(x, lag), self.dtype)


class ValidatingEvaluator(Evaluator):
    def __init__(self, problem, params):
        super().__init__(problem, params)
        self.num_vars = problem.num_vars
        self.num_cons = problem.num_cons

    def _eval_obj(self, x: np.ndarray) -> float:
        obj = self.problem.obj(x)

        if not math.isfinite(obj):
            raise EvalError("Infinite objective", x)

        return obj

    def _eval_obj_grad(self, x: np.ndarray) -> np.ndarray:
        grad = self.problem.obj_grad(x)

        if grad.shape != (self.num_vars,):
            raise EvalError("Invalid shape of gradient", x)

        if not np.isfinite(grad).all():
            raise EvalError("Non-finite gradient", x)

        return astype(grad, self.dtype)

    def _eval_cons(self, x: np.ndarray) -> np.ndarray:
        if self.num_cons == 0:
            return np.array([], dtype=self.dtype)

        cons = self.problem.cons(x)

        if cons.shape != (self.num_cons,):
            raise EvalError("Invalid shape of constraints", x)

        if not np.isfinite(cons).all():
            raise EvalError("Non-finite constraints", x)

        return astype(cons, self.dtype)

    def _eval_cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        if self.num_cons == 0:
            return sp.sparse.csr_matrix((0, self.num_vars), dtype=self.dtype)

        cons_jac = self.problem.cons_jac(x)

        if cons_jac.shape != (self.num_cons, self.num_vars):
            raise EvalError("Invalid shape of Jacobian", x)

        if not np.isfinite(cons_jac.data).all():
            raise EvalError("Non-finite Jacobian", x)

        return astype(cons_jac, self.dtype)

    def _eval_lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        lag_hess = self.problem.lag_hess(x, lag)

        if lag_hess.shape != (self.num_vars, self.num_vars):
            raise EvalError("Invalid shape of Hessian", x)

        if not np.isfinite(lag_hess.data).all():
            raise EvalError("Non-finite Hessian", x)

        coo_hess = lag_hess.tocoo()
        coo_hess_T = coo_hess.T

        orig_pattern = set(zip(coo_hess.row, coo_hess.col))
        trans_pattern = set(zip(coo_hess.col, coo_hess.row))

        surplus_orig = orig_pattern - trans_pattern
        surplus_trans = trans_pattern - orig_pattern

        same_pattern = not (surplus_orig or surplus_trans)

        if not same_pattern:
            warn_hessian_pattern()
        else:
            coo_data = coo_hess.data
            coo_T_data = coo_hess_T.data
            if not np.allclose(coo_data, coo_T_data):
                warn_hessian_values()

        return astype(lag_hess, self.dtype)


def create_evaluator(problem, params):
    if params.validate_input:
        return ValidatingEvaluator(problem, params)
    else:
        return SimpleEvaluator(problem, params)
