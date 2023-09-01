import math
import numpy as np
import pytest
from pygradflow.controller import Controller, LogController, ControllerSettings


# Control "x' = u" using PI controller to choose u.
def update_state(x, u):
    return x + u


@pytest.fixture
def controller_settings():
    return ControllerSettings(K_P=1e-1, K_I=0.0, lamb_init=0.0, lamb_red=0.5)


@pytest.mark.parametrize("val", [0.0, 1.0, 2.0])
def test_controller_initial(controller_settings, val):
    ref_val = 1.0
    controller = Controller(controller_settings, ref_val)
    lamb_next = controller.update(val)

    if val > ref_val:
        assert lamb_next < 0.0
    elif val == ref_val:
        assert lamb_next == 0.0
    else:
        assert lamb_next > 0.0


# Note: These depend on the controller settings.
@pytest.mark.parametrize("val", [0.0, 1.0, 2.0])
def test_controller_convergence(controller_settings, val):
    ref_val = 1.0
    controller = Controller(controller_settings, ref_val)

    for _ in range(100):
        control = controller.update(val)
        next_val = update_state(val, control)
        val = next_val

    assert np.allclose(val, ref_val, atol=1e-2)


@pytest.mark.parametrize("val", [1e-1, 1e0, 1e1])
def test_log_controller_initial(controller_settings, val):
    ref_val = 1e0
    controller = LogController(controller_settings, ref_val)
    control = controller.update(val)
    control = math.log(control)

    if val > ref_val:
        assert control < 0.0
    elif val == ref_val:
        assert control == 0.0
    else:
        assert control > 0.0


# Note: These depend on the controller settings.
@pytest.mark.parametrize("val", [1e-1, 1e0, 1e1, 1e2])
def test_log_controller_convergence(controller_settings, val):
    ref_val = 10.0

    # Constants need to be larger for log controller.
    controller_settings.K_P = 1e0
    controller_settings.K_I = 1e-4

    controller = LogController(controller_settings, ref_val)

    vals = []
    controls = []

    for _ in range(100):
        control = controller.update(val)
        control = math.log(control)
        next_val = update_state(val, control)
        vals.append(val)
        controls.append(control)
        val = next_val

    assert np.allclose(val, ref_val, atol=1e-2)
