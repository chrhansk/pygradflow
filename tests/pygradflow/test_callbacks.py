from pygradflow.callbacks import CallbackType
from pygradflow.iterate import Iterate
from pygradflow.solver import Solver

from .instances import hs71_instance


def test_callbacks(hs71_instance):
    problem = hs71_instance.problem
    x_0 = hs71_instance.x_0
    y_0 = hs71_instance.y_0

    all_last_x = [None]
    all_last_y = [None]
    all_num_calls = [0]

    def computed_step(iterate: Iterate, next_iterate: Iterate, accept: bool):
        all_num_calls[0] += 1
        if accept:
            all_last_x[0] = next_iterate.x
            all_last_y[0] = next_iterate.y

    solver = Solver(problem)

    solver.callbacks.register(CallbackType.ComputedStep, computed_step)

    result = solver.solve(x_0, y_0)

    last_x = all_last_x[0]
    last_y = all_last_y[0]
    num_calls = all_num_calls[0]

    assert num_calls > 0
    assert (last_x == result.x).all()
    assert (last_y == result.y).all()
