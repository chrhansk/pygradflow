import abc
import math
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


class Evaluator(abc.ABC):

    @abc.abstractmethod
    def obj(self, x: np.ndarray) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def cons(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()

    @abc.abstractmethod
    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        raise NotImplementedError()


class SimpleEvaluator(Evaluator):
    def __init__(self, problem, params):
        self.problem = problem
        self.dtype = params.dtype

    def obj(self, x: np.ndarray) -> float:
        return self.problem.obj(x)

    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        return astype(self.problem.obj_grad(x), self.dtype)

    def cons(self, x: np.ndarray) -> np.ndarray:
        return astype(self.problem.cons(x), self.dtype)

    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        return astype(self.problem.cons_jac(x), self.dtype)

    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        return astype(self.problem.lag_hess(x, lag), self.dtype)


class ValidatingEvaluator(Evaluator):
    def __init__(self, problem, params):
        self.problem = problem
        self.dtype = params.dtype

    def obj(self, x: np.ndarray) -> float:
        obj = self.problem.obj(x)

        if not math.isfinite(obj):
            raise EvalError("Infinite objective", x)

        return obj

    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        grad = self.problem.obj_grad(x)

        if grad.shape != (self.num_vars,):
            raise EvalError("Invalid shape of gradient", x)

        if not np.isfinite(grad).all():
            raise EvalError("Non-finite gradient", x)

        return astype(grad, self.dtype)

    def cons(self, x: np.ndarray) -> np.ndarray:
        cons = self.problem.cons(x)

        if cons.shape != (self.num_cons,):
            raise EvalError("Invalid shape of constraints", x)

        if not np.isfinite(cons).all():
            raise EvalError("Non-finite constraints")

        return astype(cons, self.dtype)

    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        cons_jac = self.problem.cons_jac(x)

        if cons_jac.shape != (self.num_cons, self.num_vars):
            raise EvalError("Invalid shape of Jacobian", x)

        if not np.isfinite(cons_jac.data).all():
            raise EvalError("Non-finite Jacobian")

        return astype(cons_jac, self.dtype)

    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        lag_hess = self.problem.lag_hess(x, lag)

        if lag_hess.shape != (self.num_vars, self.num_vars):
            raise EvalError("Invalid shape of Hessian", x)

        if not np.isfinite(lag_hess.data).all():
            raise EvalError("Non-finite Hessian")

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
