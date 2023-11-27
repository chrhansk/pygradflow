import numpy as np

from pygradflow.implicit_func import ImplicitFunc, ScaledImplicitFunc
from pygradflow.iterate import Iterate
from pygradflow.params import Params

from .tame import Tame


def test_convergence():
    problem = Tame()
    params = Params()
    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])
    dt = 1e-10
    rho = 1.0

    iterate = Iterate(problem, params, x_0, y_0)

    func = ImplicitFunc(problem, iterate, dt)

    assert np.allclose(func.value_at(iterate, rho=rho), 0.0)

    func = ScaledImplicitFunc(problem, iterate, dt)

    assert np.allclose(dt * func.value_at(iterate, rho=rho), 0.0)
