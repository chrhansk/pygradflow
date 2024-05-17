from enum import Enum, auto


class SolverStatus(Enum):
    Optimal = auto()
    """
    The algorithm has converged to a solution satisfying
    the optimality conditions according to given tolerances
    """

    IterationLimit = auto()
    """
    Reached the iteration limit precribed by the algorithmic
    parameters
    """

    TimeLimit = auto()
    """
    Reached the time limit precribed by the algorithmic
    parameters
    """

    Unbounded = auto()
    """
    Problem appearst unbounded (found feasible point with extremely
    small objective value)
    """

    LocallyInfeasible = auto()
    """
    Local infeasibility detected (found infeasible point being
    a local minimum with respect to constraint violation)
    """

    @staticmethod
    def short_name(status):
        return {
            SolverStatus.Optimal: "optimal",
            SolverStatus.IterationLimit: "iteration_limit",
            SolverStatus.TimeLimit: "time_limit",
            SolverStatus.Unbounded: "unbounded",
            SolverStatus.LocallyInfeasible: "infeasible",
        }[status]

    @staticmethod
    def description(status):
        return {
            SolverStatus.Optimal: "Converged to first-order optimal solution",
            SolverStatus.IterationLimit: "Reached iteration limit",
            SolverStatus.TimeLimit: "Reached time limit",
            SolverStatus.Unbounded: "Problem appears unbounded",
            SolverStatus.LocallyInfeasible: "Local infeasibility detected",
        }[status]

    @staticmethod
    def success(status):
        """
        Returns
        -------
        bool
            Whether the status indicates a successful solve
        """
        return status == SolverStatus.Optimal
