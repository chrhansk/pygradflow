from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np
from termcolor import colored

from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.problem import Problem
from pygradflow.timer import SimpleTimer


class StateData:
    def __init__(self):
        self._entries = dict()

    def __setitem__(self, key, value):
        self._entries[key] = value

    def __getitem__(self, key):
        entry = self._entries[key]
        if callable(entry):
            return entry()
        return entry


class Format:
    @staticmethod
    def bold(s: str) -> str:
        return colored(s, attrs=["bold"])

    @staticmethod
    def _cond_color(cond: bool) -> Literal["red", "green"]:
        return "green" if cond else "red"

    @staticmethod
    def redgreen(s: str, cond: bool, bold: bool) -> str:
        if bold:
            return colored(s, Format._cond_color(cond), attrs=["bold"])

        return colored(s, Format._cond_color(cond))


class BoldFormatter:
    def __init__(self, format: str):
        self.format = format

    def __call__(self, state):
        return Format.bold(self.format.format(state))


class StringFormatter:
    def __init__(self, format: str):
        self.format = format

    def __call__(self, state):
        return self.format.format(state)


class StepFormatter:
    def __call__(self, state):
        accepted = state
        accept_str = "accepted" if accepted else "rejected"
        return Format.redgreen(accept_str, accepted, bold=True)


class RCondFormatter:
    def __call__(self, state):
        if state is None:
            return "-"

        return "{:5.0e}".format(state)


class Column(ABC):
    def __init__(self, name: str, width: int):
        self.name = name
        self.width = width

    @property
    def header(self) -> str:
        return "{:^{}s}".format(self.name, self.width)

    @abstractmethod
    def content(self, state, last_state) -> str:
        raise NotImplementedError()


class AttrColumn(Column):
    def __init__(self, name: str, width: int, format, attr):
        if isinstance(format, str):
            self.format = StringFormatter(format)
        else:
            self.format = format

        super().__init__(name, width)

        self.attr = attr

    def content(self, state, _) -> str:
        return self.format(self.attr(state))


class Display:
    def __init__(self, cols, interval=None):
        self.cols = cols
        self.interval = interval

        self.timer = None
        if self.interval is not None:
            assert self.interval > 0
            self.timer = SimpleTimer()
        self.last_state = None

    def should_display(self):
        if self.timer is None:
            return True

        return self.timer.elapsed() >= self.interval

    @property
    def header(self) -> str:
        return " ".join([col.header for col in self.cols])

    def row(self, state) -> str:
        if self.timer is not None:
            self.timer.reset()

        row = " ".join([col.content(state, self.last_state) for col in self.cols])
        self.last_state = state
        return row


class StateAttr:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state):
        value = state[self.name]
        return value


class IterateAttr:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state):
        iterate = state["iterate"]
        return getattr(iterate, self.name)


class ActiveSetColumn(Column):
    def __init__(self):
        super().__init__("Active set", 10)

    def empty(self):
        return "{:^{}s}".format("--", self.width)

    def content(self, state, last_state):
        curr_active_set = state["active_set"]
        last_active_set = None

        if (last_state is not None) and (state["iter"] == (last_state["iter"] + 1)):
            last_active_set = last_state["active_set"]

        if curr_active_set is None:
            return self.empty()

        display_curr = False

        if last_active_set is None:
            display_curr = True
        elif (curr_active_set != last_active_set).any():
            display_curr = True

        if display_curr:
            num_active = curr_active_set.sum()
            return "{:^{}d}".format(num_active, self.width)
        else:
            return "{:^{}s}".format("--", self.width)


def iter_cols(problem):
    is_bounded = problem.var_bounded
    cols = []
    cols.append(AttrColumn("Aug Lag", 16, "{:16.8e}", StateAttr("aug_lag")))
    cols.append(AttrColumn("Objective", 16, "{:16.8e}", IterateAttr("obj")))

    if is_bounded:
        cols.append(
            AttrColumn("Bound inf", 16, "{:16.8e}", IterateAttr("bound_violation"))
        )

    if problem.num_cons > 0:
        cols.append(
            AttrColumn("Cons inf", 16, "{:16.8e}", IterateAttr("cons_violation"))
        )

    cols.append(AttrColumn("Dual inf", 16, "{:16.8e}", IterateAttr("stat_res")))

    return cols


def solver_display(problem: Problem, params: Params):
    cols: List[Column] = []

    cols.append(AttrColumn("Iter", 6, BoldFormatter("{:6d}"), StateAttr("iter")))

    cols += iter_cols(problem)

    cols.append(
        AttrColumn("Primal step", 16, "{:16.8e}", StateAttr("primal_step_norm"))
    )
    cols.append(AttrColumn("Dual step", 16, "{:16.8e}", StateAttr("dual_step_norm")))
    cols.append(AttrColumn("Lambda", 16, "{:16.8e}", StateAttr("lamb")))

    if problem.var_bounded:
        cols.append(ActiveSetColumn())

    if params.report_rcond:
        cols.append(AttrColumn("Rcond", 5, RCondFormatter(), StateAttr("rcond")))

    cols.append(AttrColumn("Obj nonlin", 16, "{:16.8e}", StateAttr("obj_nonlin")))

    if problem.num_cons > 0:
        cols.append(AttrColumn("Cons nonlin", 16, "{:16.8e}", StateAttr("cons_nonlin")))

    cols.append(AttrColumn("Type", 8, StepFormatter(), StateAttr("step_accept")))

    return Display(cols, interval=params.display_interval)


class FilterColumn(Column):
    def __init__(self):
        super().__init__("Filter", 10)

    def empty(self):
        return "{:^{}s}".format("--", self.width)

    def content(self, state, last_state):
        curr_filter = state["filter"]
        last_filter = None

        if (last_state is not None) and (state["iter"] == (last_state["iter"] + 1)):
            last_filter = last_state["filter"]

        if curr_filter is None:
            return self.empty()

        display_curr = False

        if last_filter is None:
            display_curr = True
        elif (curr_filter != last_filter).any():
            display_curr = True

        if display_curr:
            num_active = curr_filter.sum()
            return "{:^{}d}".format(num_active, self.width)
        else:
            return "{:^{}s}".format("--", self.width)


class StepTypeColumn(Column):
    def __init__(self):
        super().__init__("Step Type", 10)
        self.format = BoldFormatter("{:^10s}")

    def content(self, state, last_state):
        step_type = state["step_type"]
        return self.format(step_type.name())


def integrator_display(problem: Problem, params: Params):
    cols: List[Column] = []

    cols.append(AttrColumn("Iter", 6, BoldFormatter("{:6d}"), StateAttr("iter")))
    cols += iter_cols(problem)

    cols.append(FilterColumn())

    cols.append(AttrColumn("Func evals", 10, "{:10d}", StateAttr("num_func_evals")))
    cols.append(AttrColumn("Jac evals", 10, "{:10d}", StateAttr("num_jac_evals")))
    cols.append(AttrColumn("Steps", 10, "{:10d}", StateAttr("num_steps")))
    cols.append(AttrColumn("dt", 12, "{:6e}", StateAttr("dt")))

    cols.append(StepTypeColumn())

    return Display(cols)


def inner_display(problem: Problem, params: Params):
    cols: List[Column] = []

    cols.append(AttrColumn("Iter", 6, BoldFormatter("{:5d}"), StateAttr("iter")))
    cols.append(AttrColumn("Residuum", 16, "{:16.8e}", StateAttr("residuum")))
    cols.append(AttrColumn("Dist", 16, "{:16.8e}", StateAttr("dist")))
    cols.append(AttrColumn("Active set", 10, "{:10d}", StateAttr("active_set_size")))

    return Display(cols)


def print_problem_stats(problem, iterate):
    num_vars = problem.num_vars
    num_cons = problem.num_cons

    logger.info("Solving problem with %s variables, %s constraints", num_vars, num_cons)

    cons_jac = iterate.cons_jac
    lag_hess = iterate.lag_hess(iterate.y)

    var_lb = problem.var_lb
    var_ub = problem.var_ub

    inf_lb = var_lb == -np.inf
    inf_ub = var_ub == np.inf

    unbounded = np.logical_and(inf_lb, inf_ub)
    ranged = np.logical_and(np.logical_not(inf_lb), np.logical_not(inf_ub))
    bounded_below = np.logical_and(np.logical_not(inf_lb), inf_ub)
    bounded_above = np.logical_and(inf_lb, np.logical_not(inf_ub))
    bounded = np.logical_or(bounded_below, bounded_above)

    num_unbounded = np.sum(unbounded)
    num_ranged = np.sum(ranged)
    num_bounded = np.sum(bounded)

    has_cons = num_cons > 0

    logger.info("           Hessian nnz: %s", lag_hess.nnz)
    if has_cons:
        logger.info("          Jacobian nnz: %s", cons_jac.nnz)

    logger.info("        Free variables: %s", num_unbounded)
    logger.info("     Bounded variables: %s", num_bounded)
    logger.info("      Ranged variables: %s", num_ranged)

    if not has_cons:
        return

    cons_lb = problem.cons_lb
    cons_ub = problem.cons_ub

    equation = cons_lb == cons_ub
    ranged = np.logical_and(np.isfinite(cons_lb), np.isfinite(cons_ub))
    ranged = np.logical_and(ranged, np.logical_not(equation))

    num_equations = equation.sum()
    num_ranged = ranged.sum()
    num_inequalities = num_cons - num_equations - num_ranged

    logger.info("  Equality constraints: %s", num_equations)
    logger.info("Inequality constraints: %s", num_inequalities)
    logger.info("    Ranged constraints: %s", num_ranged)
