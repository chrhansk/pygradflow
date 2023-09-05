import numpy as np


class ActiveSet:
    def __init__(self, iterate):
        active_tol = iterate.params.active_tol

        lb = iterate.problem.var_lb
        ub = iterate.problem.var_ub

        at_lower = np.absolute(iterate.x - lb) <= active_tol
        at_upper = np.absolute(ub - iterate.x) <= active_tol

        violated_lower = lb - iterate.x > active_tol
        violated_upper = iterate.x - ub > active_tol

        self.violated = np.logical_or(violated_lower, violated_upper)

        self.at_either = np.logical_or(at_lower, at_upper)

        self.at_both = np.logical_and(at_upper, at_lower)
        not_at_both = np.logical_not(self.at_both)

        self.at_lower = np.logical_and(at_lower, not_at_both)
        self.at_upper = np.logical_and(at_upper, not_at_both)

    @property
    def satisfied(self):
        return np.logical_not(self.violated)
