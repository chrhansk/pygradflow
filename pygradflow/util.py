import numpy as np


def norm_sq(x):
    return np.dot(x, x)


def norm_mult(*args):
    value = 0.0
    for arg in args:
        value += norm_sq(arg)

    return np.sqrt(value)
