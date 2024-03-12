import numpy as np
import pytest

from .hs71 import HS71
from .hs71_cons import HS71Constrained
from .rosenbrock import Rosenbrock
from .tame import Tame


class Instance:
    def __init__(self, problem, x_0, y_0, x_opt, y_opt):
        self.problem = problem
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_opt = x_opt
        self.y_opt = y_opt


@pytest.fixture
def rosenbrock_instance():
    problem = Rosenbrock()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([])

    x_opt = np.array([1.0, 1.0])
    y_opt = np.array([])

    return Instance(problem, x_0, y_0, x_opt, y_opt)


@pytest.fixture
def hs71_instance():
    problem = HS71()

    x0 = np.array([1.0, 5.0, 5.0, 1.0, 0.0])
    y0 = np.array([0.0, 0.0])

    x_opt = np.array([1.0, 4.74299964, 3.82114998, 1.37940829, 0.0])
    y_opt = np.array([-0.55229366, 0.16146857])

    return Instance(problem, x0, y0, x_opt, y_opt)


@pytest.fixture
def hs71_constrained_instance():
    problem = HS71Constrained()

    x_0 = np.array([1.0, 5.0, 5.0, 1.0])
    y_0 = np.array([0.0, 0.0])

    x_opt = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    y_opt = np.array([-0.55229366, 0.16146857])

    return Instance(problem, x_0, y_0, x_opt, y_opt)


@pytest.fixture
def tame_instance():
    problem = Tame()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])

    x_opt = np.array([0.5, 0.5])
    y_opt = np.array([0.0])

    return Instance(problem, x_0, y_0, x_opt, y_opt)
