import numpy as np
from numpy import ndarray


def norm_sq(x: ndarray) -> float:
    return np.dot(x, x)


def norm_mult(*args) -> float:
    value = 0.0
    for arg in args:
        value += norm_sq(arg)

    return np.sqrt(value)
