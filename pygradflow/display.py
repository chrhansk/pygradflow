from abc import ABC, abstractmethod
from typing import List, Literal

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


class Column(ABC):
    def __init__(self, name: str, width: int):
        self.name = name
        self.width = width

    @property
    def header(self) -> str:
        return "{:^{}s}".format(self.name, self.width)

    @abstractmethod
    def content(self, state) -> str:
        raise NotImplementedError()


class AttrColumn(Column):
    def __init__(self, name: str, width: int, format, attr):
        if isinstance(format, str):
            self.format = StringFormatter(format)
        else:
            self.format = format

        super().__init__(name, width)

        self.attr = attr

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


class ActiveSetColumn(Column):
    def __init__(self):
        super().__init__("Active set", 10)

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

    cols: List[Column] = []

    cols.append(AttrColumn("Iter", 6, BoldFormatter("{:6d}"), StateAttr("iter")))
    cols.append(AttrColumn("Aug Lag", 16, "{:16.8e}", StateAttr("aug_lag")))
    cols.append(AttrColumn("Objective", 16, "{:16.8e}", IterateAttr("obj")))

    if is_bounded:
        cols.append(
            AttrColumn("Bound inf", 16, "{:16.8e}", IterateAttr("bound_violation"))
        )

    cols.append(AttrColumn("Cons inf", 16, "{:16.8e}", IterateAttr("cons_violation")))
    cols.append(AttrColumn("Dual inf", 16, "{:16.8e}", IterateAttr("stat_res")))
    cols.append(
        AttrColumn("Primal step", 16, "{:16.8e}", StateAttr("primal_step_norm"))
    )
    cols.append(AttrColumn("Dual step", 16, "{:16.8e}", StateAttr("dual_step_norm")))
    cols.append(AttrColumn("Lambda", 16, "{:16.8e}", StateAttr("lamb")))

    if problem.var_bounded:
        cols.append(ActiveSetColumn())

    if params.report_rcond:
        cols.append(AttrColumn("Rcond", 5, RCondFormatter(), StateAttr("rcond")))

    cols.append(AttrColumn("Type", 8, StepFormatter(), StateAttr("step_accept")))

    return Display(cols)


def inner_display(problem: Problem, params: Params):
    cols: List[Column] = []

    cols.append(AttrColumn("Iter", 6, BoldFormatter("{:5d}"), StateAttr("iter")))
    cols.append(AttrColumn("Residuum", 16, "{:16.8e}", StateAttr("residuum")))
    cols.append(AttrColumn("Dist", 16, "{:16.8e}", StateAttr("dist")))
    cols.append(AttrColumn("Active set", 10, "{:10d}", StateAttr("active_set_size")))

    return Display(cols)
