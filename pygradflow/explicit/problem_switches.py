from enum import Enum, auto

from pygradflow.explicit.flow import func_neg, func_pos, lazy_func


class TriggerType(Enum):
    LB = auto()
    UB = auto()
    GRAD_FIXED = auto()
    GRAD_FREE = auto()
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

        return at_lb

    def ub_event(self, j):
        ub = self.problem.var_ub

        def at_ub(_, z):
            (x, y) = self.flow.split_states(z)
            return ub[j] - x[j]

        at_ub.type = TriggerType.UB
        at_ub.index = j

        return at_ub

    def grad_fixed_event(self, j, rho):
        def grad(_, z):
            return self.flow.neg_aug_lag_deriv_x(z, rho)[j]

        grad.type = TriggerType.GRAD_FIXED
        grad.index = j

        return grad

    def grad_free_event(self, j, rho, deriv=False):

        if deriv:

            def grad(_, z):
                return self.flow.rhs_deriv_x(z, rho)[j]

        else:

            def grad(_, z):
                return self.flow.neg_aug_lag_deriv_x(z, rho)[j]

        grad.type = TriggerType.GRAD_FREE
        grad.index = j

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
        grad_free_events = []
        grad_fixed_events = []

        at_lb = self.flow.isclose(x, lb)
        at_ub = self.flow.isclose(x, ub)

        for j in range(num_vars):
            # Variable is free
            if filter[j]:

                lb_event = self.lb_event(j)
                lb_event.direction = -1.0

                ub_event = self.ub_event(j)
                ub_event.direction = 1.0

                # Between bounds: Trigger event when approaching
                # either of the bounds
                if not (at_lb[j] or at_ub[j]):
                    lb_events.append(lb_event)
                    ub_events.append(ub_event)
                    continue

                # If at bound (wlog lb): Cannot create event
                # for reaching lb, since that would trigger again
                # immediately. However: We know that (rhs, rhs') > 0
                # so we initially move away from the bound and can only
                # reach the bound again when the sign of (rhs, rhs')
                # changes back, so just watch for that happening...
                deriv = bool(self.flow.isclose(rhs()[j], 0.0))
                grad_free_event = self.grad_free_event(j, rho, deriv=deriv)

                if at_lb[j]:
                    assert func_pos(rhs, rhs_deriv, j)
                    ub_events.append(ub_event)
                    grad_free_events.append(grad_free_event)

                elif at_ub[j]:
                    assert func_neg(rhs, rhs_deriv, j)
                    lb_events.append(lb_event)
                    grad_free_events.append(grad_free_event)

            # Variable is pinned, watch for sign changes
            if not filter[j]:
                # Equal bounds, flow *must* stay at zero
                if at_lb[j] and at_ub[j]:
                    continue

                event = self.grad_fixed_event(j, rho)

                if at_lb[j]:
                    # We only care about the derivative
                    # becoming positive
                    event.direction = 1.0
                elif at_ub[j]:
                    # We only care about the derivative
                    # becoming negative
                    event.direction = -1.0

                grad_fixed_events.append(event)

        events = [
            *lb_events,
            *ub_events,
            *grad_free_events,
            *grad_fixed_events,
            self.converged_event(),
            self.unbounded_event(),
        ]

        for event in events:
            event.terminal = True

        return events
