import abc
import numpy as np


class Problem(abc.ABC):
    def __init__(self, var_lb, var_ub, num_cons=0):
        assert var_lb.shape == var_ub.shape
        assert var_lb.ndim == 1

        assert (var_lb <= var_ub).all()
        assert (var_lb < np.inf).all()
        assert (var_ub > -np.inf).all()

        self.var_lb = np.copy(var_lb)
        self.var_ub = np.copy(var_ub)
        self.num_cons = num_cons

    @property
    def num_vars(self):
        (num_vars,) = self.var_lb.shape

        return num_vars

    @abc.abstractmethod
    def obj(self, x):
        pass

    @abc.abstractmethod
    def obj_grad(self, x):
        pass

    @abc.abstractmethod
    def cons(self, x):
        pass

    @abc.abstractmethod
    def cons_jac(self, x):
        pass

    @abc.abstractmethod
    def lag_hess(self, x, lag):
        pass
