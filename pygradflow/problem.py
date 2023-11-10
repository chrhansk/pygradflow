import abc
import functools

import numpy as np
import scipy as sp


class Problem(abc.ABC):
    def __init__(
        self, var_lb: np.ndarray, var_ub: np.ndarray, num_cons: int = 0
    ) -> None:
        assert var_lb.shape == var_ub.shape
        assert var_lb.ndim == 1

        assert (var_lb <= var_ub).all()
        assert (var_lb < np.inf).all()
        assert (var_ub > -np.inf).all()

        self.var_lb = np.copy(var_lb)
        self.var_ub = np.copy(var_ub)
        self.num_cons = num_cons

    @functools.cached_property
    def var_bounded(self):
        return np.isfinite(self.var_lb).any() or np.isfinite(self.var_ub).any()

    @property
    def num_vars(self) -> int:
        (num_vars,) = self.var_lb.shape

        return num_vars

    @abc.abstractmethod
    def obj(self, x: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def cons(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        pass

    @abc.abstractmethod
    def lag_hess(self, x: np.ndarray, lag: np.ndarray) -> sp.sparse.spmatrix:
        pass
