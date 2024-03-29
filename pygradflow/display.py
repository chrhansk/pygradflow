from typing import Literal

from termcolor import colored

from pygradflow.params import Params
from pygradflow.problem import Problem


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


class Column:
    def __init__(self, name: str, width: int, format, attr):
        self.name = name
        self.width = width

        if isinstance(format, str):
            self.format = StringFormatter(format)
        else:
            self.format = format

        self.attr = attr

    @property
    def header(self) -> str:
        return "{:^{}s}".format(self.name, self.width)

    def content(self, state) -> str:
        return self.format(self.attr(state))


class Display:
    def __init__(self, cols):
        self.cols = cols

    @property
    def header(self) -> str:
        return " ".join([col.header for col in self.cols])

    def row(self, state) -> str:
        return " ".join([col.content(state) for col in self.cols])


class StateAttr:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state):
        value = state[self.name]
        return value()


class IterateAttr:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state):
        iterate = state["iterate"]
        return getattr(iterate, self.name)


class ActiveSetColumn:
    def __init__(self):
        self.width = 10
        self.name = "Active set"

    @property
    def header(self):
        return "{:^{}s}".format(self.name, self.width)

    def empty(self):
        return "{:^{}s}".format("--", self.width)

    def content(self, state):
        curr_active_set = state["curr_active_set"]()
        last_active_set = state["last_active_set"]()

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


def problem_display(problem: Problem, params: Params):
    is_bounded = problem.var_bounded

    cols = []

    cols.append(Column("Iter", 6, BoldFormatter("{:6d}"), StateAttr("iter")))
    cols.append(Column("Aug Lag", 16, "{:16.8e}", StateAttr("aug_lag")))

    if is_bounded:
        cols.append(Column("Bound inf", 16, "{:16.8e}", IterateAttr("bound_violation")))

    cols.append(Column("Cons inf", 16, "{:16.8e}", IterateAttr("cons_violation")))
    cols.append(Column("Dual inf", 16, "{:16.8e}", IterateAttr("stat_res")))
    cols.append(Column("Primal step", 16, "{:16.8e}", StateAttr("primal_step_norm")))
    cols.append(Column("Dual step", 16, "{:16.8e}", StateAttr("dual_step_norm")))
    cols.append(Column("Lambda", 16, "{:16.8e}", StateAttr("lamb")))

    if problem.var_bounded:
        cols.append(ActiveSetColumn())

    if params.report_rcond:
        cols.append(Column("Rcond", 5, RCondFormatter(), StateAttr("rcond")))

    cols.append(Column("Type", 8, StepFormatter(), StateAttr("step_accept")))

    return Display(cols)


def inner_display(problem: Problem, params: Params):
    cols = []

    cols.append(Column("Iter", 6, BoldFormatter("{:5d}"), StateAttr("iter")))
    cols.append(Column("Residuum", 16, "{:16.8e}", StateAttr("residuum")))
    cols.append(Column("Dist", 16, "{:16.8e}", StateAttr("dist")))
    cols.append(Column("Active set", 10, "{:10d}", StateAttr("active_set_size")))

    return Display(cols)
