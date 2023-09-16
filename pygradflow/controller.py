from dataclasses import dataclass
import math

from pygradflow.params import Params


@dataclass
class ControllerSettings:
    K_P: float = 0.0
    K_I: float = 0.0

    lamb_init: float = 0.0
    lamb_red: float = 0.0

    def __post_init__(self) -> None:
        assert self.K_P >= 0.0
        assert self.K_I >= 0.0

    @staticmethod
    def from_params(params: Params) -> "ControllerSettings":
        return ControllerSettings(
            K_P=params.K_P,
            K_I=params.K_I,
            lamb_init=params.lamb_init,
            lamb_red=params.lamb_red,
        )


class Controller:
    """
    PI controller with given settings and reference value.
    """

    def __init__(self, settings: ControllerSettings, ref: float) -> None:
        self.settings = settings
        self.ref = ref

        self.value = settings.lamb_init
        self.error_sum = 0.0

    def reset(self) -> None:
        self.error_sum = 0.0

    def update(self, val: float) -> float:
        error = self.ref - val
        self.error_sum += error

        update_term = self.settings.K_P * error + self.settings.K_I * self.error_sum

        self.value = update_term
        return self.value


class LogController:
    """
    PI controller working on log scale.
    """

    def __init__(self, settings: ControllerSettings, ref: float) -> None:
        self.settings = settings
        assert ref > 0.0

        self.controller = Controller(settings, math.log(ref))

        self.ref = ref
        self.error_sum = 0.0

    @property
    def value(self) -> float:
        return math.exp(self.controller.value)

    def update(self, val: float) -> float:
        assert val > 0.0

        self.controller.update(math.log(val))

        return self.value
