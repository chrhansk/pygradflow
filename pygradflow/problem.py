import abc
import functools

import numpy as np
import scipy as sp


class Problem(abc.ABC):
    """
    Base class used to formulate the problem of minimizing a smooth objective
    :math:`f : \\mathbb{R}^{n} \\to \\mathbb{R}`
    over a set of smooth
    nonlinear constraints
    :math:`c : \\mathbb{R}^{n} \\to \\mathbb{R}^{m}`
    bounded by :math:`l, u \\in \\mathbb{R}^{m}`,
    and variable bounds :math:`x_l, x_u \\in \\mathbb{R}^{n}`,
    yielding the problem
     .. math::
        \\begin{align}
            \\min_{x \\in \\mathbb{R}^{n}} \\quad & f(x) \\\\
            \\text{s.t.} \\quad & l \\leq c(x) \\leq u \\\\
                              & l^x \\leq x \\leq u^x
        \\end{align}

    The Lagrangian of this problem is given by
     .. math::
        \\mathcal{L}(x, y) = f(x) + y^{T} c(x),

    where :math:`y \\in \\mathbb{R}^{m}` is a vector of Lagrange multipliers
    """

    def __init__(self, var_lb: np.ndarray, var_ub: np.ndarray, **args) -> None:
        """
        Creates the problem

        Parameters
        ----------
        var_lb, var_ub : np.ndarray
            The variable bounds :math:`l^x, u^x \\in \\mathbb{R}^{n}`
            constraining the problem
        **args : dict, optional
            Additional arguments to pass to the problem:

            * ``cons_lb`` : np.ndarray
                The lower bounds :math:`l \\in \\mathbb{R}^{m}`
                on the constraints
            * ``cons_ub`` : np.ndarray
                The upper bounds :math:`u \\in \\mathbb{R}^{m}`
                on the constraints
            * ``num_cons`` : int
                The number of constraints (if not given by ``cons_lb`` or ``cons_ub``),
                defaults to 0 (no constraints)
        """
        assert var_lb.shape == var_ub.shape
        assert var_lb.ndim == 1

        assert (var_lb <= var_ub).all()
        assert (var_lb < np.inf).all()
        assert (var_ub > -np.inf).all()

        self.var_lb = np.copy(var_lb)
        self.var_ub = np.copy(var_ub)

        num_cons = args.get("num_cons", None)
        cons_lb = args.get("cons_lb", None)
        cons_ub = args.get("cons_ub", None)

        if cons_lb is not None or cons_ub is not None:
            assert num_cons is None

            if cons_lb is not None:
                (num_cons,) = cons_lb.shape
            else:
                (num_cons,) = cons_ub.shape

            if cons_lb is None:
                cons_lb = np.zeros((num_cons,))
            if cons_ub is None:
                cons_ub = np.zeros((num_cons,))

            assert (cons_lb <= cons_ub).all()
            assert (cons_lb < np.inf).all()
            assert (cons_ub > -np.inf).all()

        else:
            if num_cons is None:
                num_cons = 0

            cons_lb = np.zeros((num_cons,))
            cons_ub = np.zeros((num_cons,))

        self.num_cons = num_cons
        self.cons_lb = cons_lb
        self.cons_ub = cons_ub

    @functools.cached_property
    def var_bounded(self):
        """
        Whether of not the variable bounds are trivial (i.e., all infinte)
        """
        return np.isfinite(self.var_lb).any() or np.isfinite(self.var_ub).any()

    @property
    def num_vars(self) -> int:
        """
        The number of variables in the problem
        """
        (num_vars,) = self.var_lb.shape

        return num_vars

    @abc.abstractmethod
    def obj(self, x: np.ndarray) -> float:
        """
        Parameters
        ----------
        x : np.ndarray
            The primal point :math:`x \\in \\mathbb{R}^{n}` at which
            to evaluate the objective function

        Returns
        -------
        float
            The objective function value :math:`f(x)` at the given primal point :math:`x`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def obj_grad(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
           The primal point :math:`x \\in \\mathbb{R}^{n}` at which to
           evaluate the objective function gradient

        Returns
        -------
        np.ndarray
            The objective function gradient :math:`\\nabla f(x)`
            at the given primal point :math:`x`
        """
        raise NotImplementedError()

    def cons(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
           The primal point :math:`x \\in \\mathbb{R}^{n}` at which to
           evaluate the constraint function :math:`c`

        Returns
        -------
        np.ndarray
            The constraint value :math:`c(x)` at the given primal point :math:`x`
        """
        raise NotImplementedError()

    def cons_jac(self, x: np.ndarray) -> sp.sparse.spmatrix:
        """
        Parameters
        ----------
        x : np.ndarray
           The primal point :math:`x \\in \\mathbb{R}^{n}` at which to
           evaluate the constraint Jacobian :math:`J_c`

        Returns
        -------
        sp.sparse.spmatrix
            The constraint Jacobian :math:`J_c(x)` at the given primal point :math:`x`
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def lag_hess(self, x: np.ndarray, y: np.ndarray) -> sp.sparse.spmatrix:
        """
        Parameters
        ----------
        x, y : np.ndarray
           The primal / dual points :math:`x \\in \\mathbb{R}^{n}` and
           :math:`y \\in \\mathbb{R}^{m}` at which to
           evaluate the Hessian :math:`\\nabla_{xx} \\mathcal{L}(x, y)`

        Returns
        -------
        sp.sparse.spmatrix
            The matrix
            :math:`\\nabla_{xx} \\mathcal{L}(x, y) \\in \\mathbb{R}^{n \\times n}`
            at the given primal / dual points :math:`x` / :math:`y`
        """
        raise NotImplementedError()
