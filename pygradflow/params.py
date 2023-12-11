from dataclasses import dataclass
from enum import Enum, Flag, auto

import numpy as np


class NewtonType(Enum):
    """
    The Newton method to be used to solve the semi-smooth systems.
    Different methods have trade-offs in terms of computational
    complexity and convergence speed
    """

    Simplified = auto()
    """
    Simplified Newton method, using the same derivative and
    active set throughout computations (cheapest)
    """
    Full = auto()
    """
    Full Newton method, using a new derivative at each step,
    (requiring new derivative evaluations and factorizations)
    """
    ActiveSet = auto()
    """
    Newton method with fixed derivative, but changing active sets.
    Requires refactorizations but no reevaluations of the derivatives
    """


class StepSolverType(Enum):
    """
    Step solver type to be used throughout computations
    """

    Standard = auto()
    """
    Unscaled and unsymmetric
    """
    Extended = auto()
    """
    Scaled step solver with improved condition
    """
    Symmetric = auto()
    """
    Scaled solver with symmetric matrix
    """
    Asymmetric = auto()
    """
    Scaled solver with asymmetric matrix
    """


class LinearSolverType(Enum):
    """
    Linear solver to be used throughout the computations
    """

    LU = auto()
    """
    LU decomposition
    """
    MINRES = auto()
    """
    Minimal residual method (MINRES), only
    works with symmetric step solver :py:class:`pygradflow.params.StepSolverType.Symmetric`
    """
    GMRES = auto()
    """
    Generalized minimal residual method (GMRES)
    """


class StepControlType(Enum):
    Exact = auto()
    ResiduumRatio = auto()
    DistanceRatio = auto()


class PenaltyUpdate(Enum):
    Constant = auto()
    DualNorm = auto()
    DualEquilibration = auto()
    ParetoDecrease = auto()


class Precision(Enum):
    """
    Precision to be used in all calculations
    """

    Single = auto()
    """
    Single precision (32 bit)
    """
    Double = auto()
    """
    Double precision (64 bit)
    """


class DerivCheck(Flag):
    """
    How to check for derivatives
    """

    NoCheck = 0
    """
    Disable checks
    """
    CheckFirst = 1 << 0
    """
    Check first derivatives (objective gradient :math:`\\nabla_{x} f(x)` and :math:`J_c(x)`)
    """
    CheckSecond = 1 << 1
    """
    Check Hessian of Lagrangian (:math:`\\nabla_{xx} \mathcal{L}(x, y)`)
    """
    CheckAll = CheckFirst | CheckSecond


@dataclass
class Params:
    """
    Parameters used to solve a :py:class:`pygradflow.problem.Problem`
    using a :py:class:`pygradflow.solver.Solver`
    """

    rho: float = 1e2

    num_it: int = 1000

    theta_max: float = 0.9
    theta_ref: float = 0.5

    lamb_init: float = 1.0
    # Up to 1e-6 for single precision?
    lamb_min: float = 1e-12
    lamb_max: float = 1e12
    lamb_inc: float = 2.0
    lamb_red: float = 0.5

    K_P: float = 0.2
    K_I: float = 0.005

    opt_tol: float = 1e-6
    lamb_term: float = 1e-8
    active_tol: float = 1e-8

    local_infeas_tol: float = 1e-8

    newton_type: NewtonType = NewtonType.Simplified
    newton_tol: float = 1e-8

    step_control_type: StepControlType = StepControlType.DistanceRatio

    step_solver: object = None
    step_solver_type: StepSolverType = StepSolverType.Symmetric
    linear_solver_type: LinearSolverType = LinearSolverType.LU
    penalty_update: PenaltyUpdate = PenaltyUpdate.DualNorm

    deriv_check: DerivCheck = DerivCheck.NoCheck
    deriv_pert: float = 1e-8
    deriv_tol: float = 1e-4

    precision: Precision = Precision.Double

    validate_input: bool = True

    time_limit: float = np.inf
    display_interval: float = 0.1

    # lower bound on objective function value,
    # used to detect unbounded problems
    obj_lower_limit: float = -1e10

    report_rcond: bool = False

    @property
    def dtype(self):
        return np.float32 if self.precision == Precision.Single else np.float64
