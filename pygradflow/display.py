from termcolor import colored

from pygradflow.params import Params
from pygradflow.problem import Problem


class Format:
    @staticmethod
    def bold(s: str) -> str:
        return colored(s, attrs=["bold"])

    @staticmethod
    def redgreen(s: str, cond: bool, bold: bool) -> str:
        color = "green" if cond else "red"

        if bold:
            return colored(s, color, attrs=["bold"])

        return colored(s, color)


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


class Column:
    def __init__(self,
                 name: str,
                 width: int,
                 format,
                 attr):
        self.name = name
        self.width = width

        if isinstance(format, str):
            self.format = StringFormatter(format)
        else:
            self.format = format

        self.attr = attr

    @property
    def header(self):
        return "{:^{}s}".format(self.name, self.width)

    def content(self, state):
        return self.format(self.attr(state))


class Display:
    def __init__(self, cols):
        self.cols = cols

    @property
    def header(self):
        return " ".join([col.header for col in self.cols])

    def row(self, state):
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

    if params.report_rcond:
        cols.append(Column("Rcond", 5, "{:5.0e}", StateAttr("rcond")))

    cols.append(Column("Type", 8, StepFormatter(), StateAttr("step_accept")))

    return Display(cols)
