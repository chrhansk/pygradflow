import numpy as np

from pygradflow.iterate import Iterate
from pygradflow.params import Params

from .tame import Tame

rho = 1.0


def test_aug_lag_deriv_xy():
    problem = Tame()
    params = Params()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])

    iterate = Iterate(problem, params, x_0, y_0)

    expected = iterate.aug_lag_deriv_xy()

    curr_deriv = iterate.aug_lag_deriv_x(rho=rho)

    x_p = np.copy(x_0)
    y_p = np.copy(y_0)
    y_p[0] += 1e-8

    next_iterate = Iterate(problem, params, x_p, y_p)

    next_deriv = next_iterate.aug_lag_deriv_x(rho=rho)

    diff = next_deriv - curr_deriv
    diff /= 1e-8

    assert np.allclose(diff, expected.toarray())


def test_aug_lag_deriv_x():
    problem = Tame()
    params = Params()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])

    iterate = Iterate(problem, params, x_0, y_0)

    cur_val = iterate.aug_lag(rho=rho)

    expected = iterate.aug_lag_deriv_x(rho=rho)

    n = x_0.shape[0]

    actual = np.zeros((n,))

    for i in range(n):
        x_n = np.copy(x_0)
        x_n[i] += 1e-8

        next_iterate = Iterate(problem, params, x_n, y_0)
        next_val = next_iterate.aug_lag(rho=rho)

        diff = next_val - cur_val
        diff /= 1e-8

        actual[i] = diff

    assert np.allclose(actual, expected)


def test_aug_lag_deriv_xx():
    problem = Tame()
    params = Params()

    x_0 = np.array([0.0, 0.0])
    y_0 = np.array([0.0])

    iterate = Iterate(problem, params, x_0, y_0)

    cur_val = iterate.aug_lag_deriv_x(rho=rho)

    expected = iterate.aug_lag_deriv_xx(rho=rho)

    n = x_0.shape[0]

    actual = np.zeros((n, n))

    for i in range(n):
        x_n = np.copy(x_0)
        x_n[i] += 1e-8

        next_iterate = Iterate(problem, params, x_n, y_0)
        next_val = next_iterate.aug_lag_deriv_x(rho=rho)

        diff = next_val - cur_val
        diff /= 1e-8

        actual[:, i] = diff

    assert np.allclose(actual, expected.toarray())
