from enum import Enum, Flag, auto
from dataclasses import dataclass

import numpy as np


class NewtonType(Enum):
    Simplified = auto()
    Full = auto()
    ActiveSet = auto()


class StepSolverType(Enum):
    Standard = auto()
    Extended = auto()
    Symmetric = auto()


class LinearSolverType(Enum):
    LU = auto()
    MINRES = auto()
    GMRES = auto()


class StepControlType(Enum):
    Exact = auto()
    ResiduumRatio = auto()
    DistanceRatio = auto()


class PenaltyUpdate(Enum):
    Constant = auto()
    DualNorm = auto()
    DualEquilibration = auto()


class Precision(Enum):
    Single = auto()
    Double = auto()


class DerivCheck(Flag):
    NoCheck = 0
    CheckFirst = (1 << 0)
    CheckSecond = (1 << 1)
    CheckAll = CheckFirst | CheckSecond


@dataclass
class Params:
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
