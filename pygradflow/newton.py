import abc
import numpy as np
import scipy as sp
from pygradflow.implicit_func import ImplicitFunc
from pygradflow.iterate import Iterate

from pygradflow.log import logger as lgg
from pygradflow.params import NewtonType
from pygradflow.step import step_solver


logger = lgg.getChild("newton")


class _NewtonMethod(abc.ABC):
    def __init__(self, problem, orig_iterate, dt, rho):
        self.problem = problem
        self.orig_iterate = orig_iterate
        self.dt = dt
        self.rho = rho
        self.func = ImplicitFunc(problem, orig_iterate, dt)

    @property
    def params(self):
        return self.orig_iterate.params

    @abc.abstractmethod
    def step(self, iterate):
        raise NotImplementedError()


class SimpleNewtonMethod(_NewtonMethod):
    """
    Computes step based on the matrix given in terms of the *initial*
    iterate. Only requires a back-solve for each step.
    """

    def __init__(self, problem, orig_iterate, dt, rho, step_solver):
        super().__init__(problem, orig_iterate, dt, rho)

        self.step_solver = step_solver
        p = self.func.projection_initial(orig_iterate, rho)
        active_set = self.func.compute_active_set(p)

        self.step_solver.update_active_set(active_set)
        self.step_solver.update_derivs(orig_iterate)

    def step(self, iterate):
        (xn, yn) = self.step_solver.solve(iterate)

        return Iterate(self.problem, self.params, xn, yn)


class FullNewtonMethod(_NewtonMethod):
    """
    Computes step based on the matrix given in terms of the *current*
    iterate. Requires evaluation of the derivative and a factorization
    at each step.
    """

    def __init__(self, problem, orig_iterate, dt, rho, step_solver):
        super().__init__(problem, orig_iterate, dt, rho)
        self.step_solver = step_solver

    def step(self, iterate):
        p = self.func.projection_initial(iterate, self.rho)
        active_set = self.func.compute_active_set(p)

        self.step_solver.update_active_set(active_set)
        self.step_solver.update_derivs(iterate)

        (xn, yn) = self.step_solver.solve(iterate)

        return Iterate(self.problem, self.params, xn, yn)


class FixedActiveSetNewtonMethod(_NewtonMethod):
    """
    Computes step based on the matrix given in terms of the *current*
    iterate. Requires evaluation of the derivative and a factorization
    at each step.
    """

    def __init__(self, problem, active_set, orig_iterate, dt, rho):
        super().__init__(problem, orig_iterate, dt, rho)

        assert active_set.dtype == bool
        assert active_set.shape == problem.var_lb.shape

        self.active_set = active_set

        logger.info(
            "Active set fingerprint: %s, size: %s",
            hex(abs(hash(active_set.data.tobytes()))),
            np.sum(active_set),
        )

    def split_sol(self, s):
        n = self.problem.num_vars
        m = self.problem.num_cons

        assert s.shape == (n + m,)
        return s[:n], s[n:]

    def create_iterate(self, iterate, s):
        xn, yn = self.split_sol(s)
        x = iterate.x
        y = iterate.y

        return Iterate(self.problem, self.params, x - xn, y - yn)

    @staticmethod
    def active_set_from_iterate(problem, iterate):
        lb = problem.var_lb
        ub = problem.var_ub

        x = iterate.x
        d = iterate.aug_lag_deriv_x()
        d_zero = np.isclose(d, 0.0)

        active_set = np.full(lb.shape, fill_value=True, dtype=bool)

        at_lower = np.isclose(x, lb)
        viol_lower = np.logical_and(x <= lb, np.logical_not(at_lower))
        active_set[viol_lower] = False

        d_neg = np.logical_and(d <= 0.0, np.logical_not(d_zero))
        active_set[np.logical_and(d_neg, at_lower)] = False

        at_upper = np.isclose(x, ub)
        viol_upper = np.logical_and(x >= ub, np.logical_not(at_upper))
        active_set[viol_upper] = False

        d_pos = np.logical_and(d >= 0.0, np.logical_not(d_zero))
        active_set[np.logical_and(d_pos, at_upper)] = False

        return active_set

    def step(self, iterate):
        rhs = self.func.value_at(iterate, self.rho, active_set=self.active_set)
        mat = self.func.deriv_at(iterate, self.rho, active_set=self.active_set)

        # logger.info(
        #     "Condition of system matrix: {0:.1e}".format(np.linalg.cond(mat.toarray()))
        # )

        solver = sp.sparse.linalg.splu(mat)

        s = solver.solve(rhs)
        next_iterate = self.create_iterate(iterate, s)

        logger.info(
            "Initial rhs norm: {0}, final: {1}".format(
                np.linalg.norm(rhs), np.linalg.norm(self.func.value_at(next_iterate))
            )
        )

        return next_iterate


class ActiveSetNewtonMethod(_NewtonMethod):
    """
    Computes step based on the matrix given in terms of the
    *initial* iterate with an active set based on the active set
    projection of the primal point of the *current* iterate. Requires a
    factorization at each step.
    """

    def __init__(self, problem, orig_iterate, dt, rho, step_solver):
        super().__init__(problem, orig_iterate, dt, rho)

        self.step_solver = step_solver
        self.step_solver.update_derivs(orig_iterate)

    def step(self, iterate):
        p = self.func.projection_initial(iterate, self.rho)
        active_set = self.func.compute_active_set(p)

        self.step_solver.update_active_set(active_set)

        (xn, yn) = self.step_solver.solve(iterate)

        return Iterate(self.problem, self.params, xn, yn)


def newton_method(problem, params, iterate, dt, rho):
    assert dt > 0.0
    assert rho > 0.0

    solver = step_solver(problem, params, iterate, dt, rho)

    if params.newton_type == NewtonType.Simple:
        return SimpleNewtonMethod(problem, iterate, dt, rho, solver)
    elif params.newton_type == NewtonType.Full:
        return FullNewtonMethod(problem, iterate, dt, rho, solver)
    elif params.newton_type == NewtonType.ActiveSet:
        return ActiveSetNewtonMethod(problem, iterate, dt, rho, solver)
