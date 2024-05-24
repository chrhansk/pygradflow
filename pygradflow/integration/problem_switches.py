from enum import Enum, auto

import numpy as np

from pygradflow.integration.flow import lazy_func


class TriggerType(Enum):
    LB = auto()
    UB = auto()
    PENALTY = auto()
    GRAD_FIXED = auto()
    CONVERGED = auto()
    UNBOUNDED = auto()


class SwitchTrigger:
    def __init__(self, time, state, trigger):
        self.time = time
        self.state = state
        self.trigger = trigger

    @property
    def type(self):
        return self.trigger.type

    @property
    def index(self):
        return self.trigger.index


class ProblemSwitches:
    def __init__(self, restricted_flow):
        self.restricted_flow = restricted_flow
        self.flow = restricted_flow.flow
        self.problem = restricted_flow.problem
        self.params = restricted_flow.params

    def lb_event(self, j):
        lb = self.problem.var_lb

        def at_lb(_, z):
            (x, y) = self.flow.split_states(z)
            return x[j] - lb[j]

        at_lb.type = TriggerType.LB
        at_lb.index = j
        at_lb.direction = -1.0

        return at_lb

    def ub_event(self, j):
        ub = self.problem.var_ub

        def at_ub(_, z):
            (x, y) = self.flow.split_states(z)
            return x[j] - ub[j]

        at_ub.type = TriggerType.UB
        at_ub.index = j
        at_ub.direction = 1.0

        return at_ub

    def penalty_event(self, rho):
        def penalty_check(_, z):
            rhs = self.restricted_flow.rhs(z, rho)
            rhs_x, rhs_y = self.flow.split_states(rhs)
            aug_dx = self.flow.aug_lag_deriv_x(z, rho)
            aug_dy = self.flow.aug_lag_deriv_y(z, rho)

            return np.dot(rhs_x, aug_dx) + np.dot(rhs_y, aug_dy)

        penalty_check.type = TriggerType.PENALTY
        penalty_check.direction = 1.0

        return penalty_check

    def grad_fixed_event(self, j, rho, at_lb):
        def grad(_, z):
            return self.flow.neg_aug_lag_deriv_x(z, rho)[j]

        grad.type = TriggerType.GRAD_FIXED
        grad.index = j

        if at_lb:
            grad.direction = 1.0
        else:
            grad.direction = -1.0

        return grad

    def converged_event(self):
        params = self.params

        def converged_event(_, z):
            return self.restricted_flow.residuum(z) - params.opt_tol

        converged_event.type = TriggerType.CONVERGED

        return converged_event

    def unbounded_event(self):
        params = self.params

        def unbounded_event(_, z):
            return self.flow.obj(z) - params.obj_lower_limit

        unbounded_event.type = TriggerType.UNBOUNDED

        return unbounded_event

    def create_event_triggers(self, filter, z_curr, rho):
        lb = self.problem.var_lb
        ub = self.problem.var_ub

        problem = self.problem
        num_vars = problem.num_vars

        assert filter.shape == (num_vars,)
        assert filter.dtype == bool

        @lazy_func
        def rhs():
            return self.flow.rhs(z_curr, rho)

        @lazy_func
        def rhs_deriv():
            return self.flow.rhs_deriv_x(z_curr, rho)

        (x, y) = self.flow.split_states(z_curr)

        lb_events = []
        ub_events = []
        grad_fixed_events = []

        at_lb = self.flow.isclose(x, lb)
        at_ub = self.flow.isclose(x, ub)

        for j in range(num_vars):
            # Variable is free
            if filter[j]:

                lb_event = self.lb_event(j)
                ub_event = self.ub_event(j)

                if np.isfinite(lb[j]):
                    lb_events.append(lb_event)

                if np.isfinite(ub[j]):
                    ub_events.append(ub_event)

            # Variable is pinned, watch for sign changes
            if not filter[j]:
                # Equal bounds, flow *must* stay at zero
                if at_lb[j] and at_ub[j]:
                    continue

                assert at_lb[j] or at_ub[j]

                event = self.grad_fixed_event(j, rho, at_lb[j])
                grad_fixed_events.append(event)

        events = [
            *lb_events,
            *ub_events,
            *grad_fixed_events,
            self.converged_event(),
            self.unbounded_event(),
            self.penalty_event(rho),
        ]

        for event in events:
            event.terminal = True

        return events
