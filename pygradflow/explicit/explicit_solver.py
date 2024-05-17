import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import scipy as sp

from pygradflow.cons_problem import ConstrainedProblem
from pygradflow.eval import create_evaluator
from pygradflow.explicit.events import (
    ConvergedResult,
    EventResultType,
    FilterChangedResult,
    FreeGradZeroResult,
    UnboundedResult,
)
from pygradflow.explicit.flow import Flow, func_neg, func_pos, lazy_func
from pygradflow.explicit.problem_switches import SwitchTrigger, TriggerType
from pygradflow.explicit.restricted_flow import RestrictedFlow
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.result import SolverResult
from pygradflow.status import SolverStatus


class IntegrationStatus(Enum):
    Converged = auto()
    Unbounded = auto()
    Event = auto()
    Finished = auto()


class IntegrationResult:
    def __init__(self, status, t_next, z_next, filter_next):
        self.status = status
        self.t = t_next
        self.z = z_next
        self.filter = filter_next


class ExplicitSolver:
    def __init__(self, problem, params):
        self.orig_problem = problem
        self.problem = ConstrainedProblem(problem)
        self.params = params

    def create_events(self, result, triggers):
        all_events = []

        for i, trigger in enumerate(triggers):
            t_events = result.t_events[i]
            z_events = result.y_events[i]

            for t, z in zip(t_events, z_events):
                all_events.append(SwitchTrigger(t, z, trigger))

        all_events.sort(key=lambda e: e.time)
        return all_events

    def create_filter(self, z, rho):
        problem = self.problem
        x = z[: problem.num_vars]
        lb = problem.var_lb
        ub = problem.var_ub

        at_lb = Flow.isclose(x, lb)
        at_ub = Flow.isclose(x, ub)
        at_bounds = np.logical_or(at_lb, at_ub)

        dx = self.flow.neg_aug_lag_deriv_x(z, rho)

        active_lower = np.logical_and(at_lb, dx < 0)
        active_upper = np.logical_and(at_ub, dx > 0)

        fixed_indices = np.logical_or(active_lower, active_upper)

        dx_zero = Flow.isclose(dx, 0.0)
        ambiguous = np.logical_and(dx_zero, at_bounds)

        if not (ambiguous.any()):
            return np.logical_not(fixed_indices)

        ddx = self.flow.rhs_deriv_x(z, rho)
        if Flow.isclose(ddx[ambiguous], 0.0).any():
            raise Exception("Degenerate bound")

        ambiguous_lb = np.logical_and(at_lb, dx_zero)
        fixed_indices[ambiguous_lb] = ddx[ambiguous_lb] < 0

        ambiguous_ub = np.logical_and(at_ub, dx_zero)
        fixed_indices[ambiguous_ub] = ddx[ambiguous_lb] > 0

        return np.logical_not(fixed_indices)

    def _check_bounds(self, z):
        problem = self.problem
        x = z[: problem.num_vars]

        lb = problem.var_lb
        ub = problem.var_ub

        between = np.logical_and(lb <= x, x <= ub)
        at_lb = Flow.isclose(x, lb)
        at_ub = Flow.isclose(x, ub)

        valid = np.logical_or(between, np.logical_or(at_lb, at_ub))

        assert valid.all()

    def _check_filter(self, z, filter, rho):
        assert (filter == self.create_filter(z, rho)).all()

    def handle_events(self, events, restricted_flow, rho):
        problem = self.problem
        lb = problem.var_lb
        ub = problem.var_ub
        filter = restricted_flow.filter
        flow = restricted_flow.flow

        logger.debug("Handling %d events", len(events))

        for event in events:
            z_event = event.state
            t_event = event.time

            self._check_bounds(z_event)

            @lazy_func
            def rhs():
                return flow.rhs(z_event, rho)

            @lazy_func
            def rhs_deriv():
                return flow.rhs_deriv_x(z_event, rho)

            if event.type == TriggerType.LB:
                j = event.index
                logger.debug("State %d reached lower bound at time %f", j, event.time)

                assert Flow.isclose(z_event[j], lb[j])

                if func_pos(rhs, rhs_deriv, j):
                    continue

                return FilterChangedResult(t_event, z_event, filter, j)

            elif event.type == TriggerType.UB:
                j = event.index
                logger.debug("State %d reached upper bound at time %f", j, event.time)

                assert Flow.isclose(z_event[j], ub[j])

                if func_neg(rhs, rhs_deriv, j):
                    continue

                return FilterChangedResult(t_event, z_event, filter, j)

            elif event.type == TriggerType.GRAD_FIXED:
                j = event.index
                logger.debug(
                    "Gradient entry of fixed %d changed sign at time %f", j, event.time
                )

                at_lb = Flow.isclose(z_event[j], lb[j])
                at_ub = Flow.isclose(z_event[j], ub[j])

                assert not (filter[j])

                if at_lb and func_neg(rhs, rhs_deriv, j):
                    continue
                elif at_ub and func_pos(rhs, rhs_deriv, j):
                    continue

                return FilterChangedResult(t_event, z_event, filter, j)

            elif event.type == TriggerType.GRAD_FREE:
                j = event.index
                logger.debug(
                    "Gradient entry of free %d changed sign at time %f", j, event.time
                )

                assert filter[j]

                at_lb = Flow.isclose(z_event[j], lb[j])
                at_ub = Flow.isclose(z_event[j], ub[j])

                assert not (at_lb or at_ub), "Not implemented"

                return FreeGradZeroResult(t_event, z_event, j)

            elif event.type == TriggerType.UNBOUNDED:
                params = self.params
                (x_event, y_event) = self.flow.split_states(z_event)
                iterate_event = Iterate(problem, params, x_event, y_event)

                if iterate_event.is_feasible(params.opt_tol):
                    return UnboundedResult(t_event, z_event)

            else:
                assert event.type == TriggerType.CONVERGED
                logger.debug("Convergence achieved at time %f", event.time)
                return ConvergedResult(t_event, z_event)

        return None

    def _get_initial_sol(self, x0, y0):
        problem = self.problem
        orig_problem = self.orig_problem

        if x0 is None:
            orig_n = orig_problem.num_vars
            x0 = np.zeros((orig_n,))
            x0 = np.clip(x0, orig_problem.var_lb, orig_problem.var_ub)

        if y0 is None:
            orig_m = orig_problem.num_cons
            y0 = np.zeros((orig_m,))

        return problem.transform_sol(x0, y0)

    def perform_integration(
        self, curr_t, curr_z, curr_filter, rho
    ) -> IntegrationResult:
        # We want to integrate to +oo...
        next_t = curr_t + 1e10

        restricted_flow = RestrictedFlow(self.flow, curr_filter)
        event_triggers = restricted_flow.create_event_triggers(curr_z, rho)

        # Check consistency
        (curr_x, curr_y) = self.flow.split_states(curr_z)
        assert self.flow.is_boxed(curr_x)
        restricted_flow.check_point(curr_z, rho)

        ivp_result = sp.integrate.solve_ivp(
            restricted_flow.rhs_func(rho),
            (curr_t, next_t),
            curr_z,
            method="BDF",
            events=event_triggers,
        )

        assert ivp_result.success, "Failed integration"

        events = self.create_events(ivp_result, event_triggers)
        event_result = self.handle_events(events, restricted_flow, rho)

        status = IntegrationStatus.Finished
        next_t = ivp_result.t[-1]
        next_z = ivp_result.y[:, -1]
        next_filter = curr_filter

        if event_result is None:
            pass
        elif event_result.type == EventResultType.CONVERGED:
            logger.debug("Convergence achieved")
            status = IntegrationStatus.Converged
            next_z = event_result.z
            next_t = event_result.t
        elif event_result.type == EventResultType.UNBOUNDED:
            logger.debug("Unboundedness detected")
            status = IntegrationStatus.Unbounded
            next_z = event_result.z
            next_t = event_result.t
        elif event_result.type == EventResultType.FILTER_CHANGED:
            logger.debug("Filter changed")
            status = IntegrationStatus.Event
            next_z = event_result.z
            next_t = event_result.t
            next_filter = event_result.filter
        elif event_result.type == EventResultType.FREE_GRAD_ZERO:
            logger.debug("Free gradient entry %d became zero", event_result.j)
            status = IntegrationStatus.Event
            next_z = event_result.z
            next_t = event_result.t

        (next_x, next_y) = self.flow.split_states(next_z)

        not_filter = np.logical_not(curr_filter)

        assert np.isclose(next_x[not_filter], curr_x[not_filter]).all()
        assert self.flow.is_boxed(next_x)

        return IntegrationResult(status, next_t, next_z, next_filter)

    def solve(self, x0: Optional[np.ndarray] = None, y0: Optional[np.ndarray] = None):
        problem = self.problem
        params = self.params

        self.eval = create_evaluator(problem, params)
        self.flow = Flow(problem, params, self.eval)

        rho = 1e2

        (x0, y0) = self._get_initial_sol(x0, y0)
        curr_z = np.concatenate((x0, y0))
        curr_t = 0.0
        curr_filter = self.create_filter(curr_z, rho)

        initial_iterate = Iterate(problem, params, x0, y0, self.eval)

        status = None
        iteration = 0
        start_time = time.time()
        path_dist = 0.0
        accepted_steps = 0

        while True:
            self._check_filter(curr_z, curr_filter, rho)
            self._check_bounds(curr_z)

            restricted_flow = RestrictedFlow(self.flow, curr_filter)
            curr_res = restricted_flow.residuum(curr_z)

            if curr_res <= params.opt_tol:
                logger.debug("Iterate is optimal")
                status = SolverStatus.Optimal
                break

            curr_time = time.time()

            logger.debug("Iteration %d: value: %s", iteration, curr_z)

            logger.debug(
                "State: %s, filter: %s, grad: %s",
                curr_z,
                curr_filter,
                self.flow.rhs(curr_z, rho),
            )

            if curr_time - start_time >= params.time_limit:
                logger.debug("Reached time limit")
                status = SolverStatus.TimeLimit
                break

            (curr_x, curr_y) = self.flow.split_states(curr_z)

            curr_it = Iterate(problem, params, curr_x, curr_y)

            if curr_it.locally_infeasible(params.opt_tol, params.local_infeas_tol):
                logger.debug("Local infeasibility detected")
                status = SolverStatus.LocallyInfeasible
                break

            if (curr_it.obj <= params.obj_lower_limit) and (
                curr_it.is_feasible(params.opt_tol)
            ):
                logger.debug("Unboundedness detected")
                status = SolverStatus.Unbounded
                break

            result = self.perform_integration(curr_t, curr_z, curr_filter, rho)
            iteration += 1

            if result.status == IntegrationStatus.Converged:
                curr_z = result.z
                status = SolverStatus.Optimal
                break
            elif result.status == IntegrationStatus.Unbounded:
                curr_z = result.z
                status = SolverStatus.Unbounded
                break

            if (params.iteration_limit is not None) and (
                iteration >= params.iteration_limit
            ):
                status = SolverStatus.IterationLimit
                logger.debug("Iteration limit reached")
                break

            curr_z = result.z
            curr_t = result.t
            curr_filter = result.filter

        (curr_x, curr_y) = self.flow.split_states(curr_z)
        iterate = Iterate(problem, params, curr_x, curr_y)

        x = iterate.x
        y = iterate.y
        d = iterate.bounds_dual

        iterations = iteration
        curr_time = time.time()
        total_time = curr_time - start_time

        logger.debug("Status: %s", status)

        direct_dist = iterate.dist(initial_iterate)

        dist_factor = path_dist / direct_dist if direct_dist != 0.0 else 1.0

        (x, y, d) = problem.restore_sol(x, y, d)

        return SolverResult(
            x,
            y,
            d,
            status,
            iterations=iterations,
            num_accepted_steps=accepted_steps,
            total_time=total_time,
            dist_factor=dist_factor,
        )
