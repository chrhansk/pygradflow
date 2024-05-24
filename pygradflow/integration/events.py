from enum import Enum, auto

import numpy as np


class EventResultType(Enum):
    CONVERGED = auto()
    UNBOUNDED = auto()
    FILTER_CHANGED = auto()
    FREE_GRAD_ZERO = auto()
    PENALTY = auto()


class EventResult:
    def __init__(self, t: float, z: np.ndarray):
        self.t = t
        self.z = z


class ConvergedResult(EventResult):
    def __init__(self, t: float, z: np.ndarray):
        super().__init__(t, z)
        self.type = EventResultType.CONVERGED


class UnboundedResult(EventResult):
    def __init__(self, t: float, z: np.ndarray):
        super().__init__(t, z)
        self.type = EventResultType.UNBOUNDED


class FilterChangedResult(EventResult):
    def __init__(self, t: float, z: np.ndarray, filter: np.ndarray, j: int):
        super().__init__(t, z)
        self.type = EventResultType.FILTER_CHANGED

        next_filter = np.copy(filter)
        (size,) = filter.shape
        assert 0 <= j < size
        next_filter[j] = not (filter[j])
        self.filter = next_filter


class PenaltyResult(EventResult):
    def __init__(self, t: float, z: np.ndarray):
        super().__init__(t, z)
        self.type = EventResultType.PENALTY
