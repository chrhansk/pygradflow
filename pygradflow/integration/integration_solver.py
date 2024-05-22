import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import scipy as sp

from pygradflow.display import StateData, integrator_display, print_problem_stats
from pygradflow.integration.events import (
    ConvergedResult,
    EventResultType,
    FilterChangedResult,
    FreeGradZeroResult,
    PenaltyResult,
    UnboundedResult,
)
from pygradflow.integration.flow import Flow, func_neg, func_pos, lazy_func
from pygradflow.integration.problem_switches import SwitchTrigger, TriggerType
from pygradflow.integration.restricted_flow import RestrictedFlow
from pygradflow.iterate import Iterate
from pygradflow.log import logger
from pygradflow.result import SolverResult
from pygradflow.status import SolverStatus
from pygradflow.timer import Timer
from pygradflow.transform import Transformation


class IntegrationStatus(Enum):
    Converged = auto()
    Unbounded = auto()
    Event = auto()
    Finished = auto()
    Penalty = auto()

    def name(self):
        return {
            IntegrationStatus.Converged: "Converged",
            IntegrationStatus.Unbounded: "Unbounded",
            IntegrationStatus.Event: "Event",
            IntegrationStatus.Finished: "Finished",
            IntegrationStatus.Penalty: "Penalty",
        }[self]


class IntegrationResult:
    def __init__(
        self,
        status,
        t_next,
        z_next,
        filter_next,
        num_steps,
        num_func_evals,
        num_jac_evals,
    ):
        self.status = status
        self.num_steps = num_steps
        self.num_func_evals = num_func_evals
        self.num_jac_evals = num_jac_evals
        self.t = t_next
        self.z = z_next
        self.filter = filter_next


class IntegrationSolver:
    def __init__(self, problem, params):
        self.orig_problem = problem
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

            elif event.type == TriggerType.PENALTY:
                (x_event, y_event) = self.flow.split_states(z_event)
                return PenaltyResult(t_event, z_event)
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
            jac=restricted_flow.rhs_jac_func(rho),
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
        else:
            assert event_result.type == EventResultType.PENALTY
            logger.debug("Penalty event")
            status = IntegrationStatus.Penalty
            next_z = event_result.z
            next_t = event_result.t

        (next_x, next_y) = self.flow.split_states(next_z)

        not_filter = np.logical_not(curr_filter)

        assert np.isclose(next_x[not_filter], curr_x[not_filter]).all()
        assert self.flow.is_boxed(next_x)

        return IntegrationResult(
            status,
            next_t,
            next_z,
            next_filter,
            num_steps=ivp_result.t.size,
            num_func_evals=ivp_result.nfev,
            num_jac_evals=ivp_result.njev,
        )

    def solve(self, x0: Optional[np.ndarray] = None, y0: Optional[np.ndarray] = None):

        self.transform = Transformation(self.orig_problem, self.params, x0, y0)

        self.problem = self.transform.trans_problem
        self.eval = self.transform.evaluator

        problem = self.problem
        params = self.params
        self.flow = Flow(problem, params, self.eval)
        rho = self.params.rho

        initial_iterate = self.transform.initial_iterate

        print_problem_stats(problem, initial_iterate)

        x_init = initial_iterate.x
        y_init = initial_iterate.y

        curr_z = np.concatenate((x_init, y_init))
        curr_t = 0.0
        curr_filter = self.create_filter(curr_z, rho)

        status = None
        iteration = 0
        path_dist = 0.0
        accepted_steps = 0

        timer = Timer(params.time_limit)

        display = integrator_display(problem, params)
        logger.info(display.header)

        while True:
            self._check_filter(curr_z, curr_filter, rho)
            self._check_bounds(curr_z)

            restricted_flow = RestrictedFlow(self.flow, curr_filter)
            curr_res = restricted_flow.residuum(curr_z)

            if curr_res <= params.opt_tol:
                logger.debug("Iterate is optimal")
                status = SolverStatus.Optimal
                break

            logger.debug("Iteration %d: value: %s", iteration, curr_z)

            logger.debug(
                "State: %s, filter: %s, grad: %s",
                curr_z,
                curr_filter,
                self.flow.rhs(curr_z, rho),
            )

            if timer.reached_time_limit():
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

            if display.should_display():
                state = StateData()
                curr_x, curr_y = self.flow.split_states(curr_z)
                iterate = Iterate(problem, params, curr_x, curr_y, self.eval)
                state["iterate"] = iterate
                state["filter"] = curr_filter
                state["aug_lag"] = lambda: iterate.aug_lag(rho)
                state["obj"] = lambda: iterate.obj()
                state["iter"] = iteration + 1
                state["num_func_evals"] = result.num_func_evals
                state["num_jac_evals"] = result.num_jac_evals
                state["num_steps"] = result.num_steps
                state["dt"] = result.t - curr_t
                state["step_type"] = result.status

                logger.info(display.row(state))

            iteration += 1

            curr_z = result.z
            curr_t = result.t
            curr_filter = result.filter

            if result.status == IntegrationStatus.Converged:
                status = SolverStatus.Optimal
                break
            elif result.status == IntegrationStatus.Unbounded:
                status = SolverStatus.Unbounded
                break
            elif result.status == IntegrationStatus.Penalty:
                # Continuation criterion is violated => update penalty parameter
                logger.debug("Updating penalty parameter %f -> %f", rho, 10 * rho)
                rho *= 10
                curr_filter = self.create_filter(curr_z, rho)

            if (params.iteration_limit is not None) and (
                iteration >= params.iteration_limit
            ):
                status = SolverStatus.IterationLimit
                logger.debug("Iteration limit reached")
                break

        (curr_x, curr_y) = self.flow.split_states(curr_z)
        iterate = Iterate(problem, params, curr_x, curr_y)

        x = iterate.x
        y = iterate.y
        d = iterate.bounds_dual

        iterations = iteration

        logger.debug("Status: %s", status)

        direct_dist = iterate.dist(initial_iterate)

        total_time = timer.elapsed()

        dist_factor = path_dist / direct_dist if direct_dist != 0.0 else 1.0

        (x, y, d) = self.transform.restore_sol(x, y, d)

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
