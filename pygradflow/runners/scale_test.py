import logging

from pygradflow.log import logger
from pygradflow.params import Params, ScalingType, StepControlType
from pygradflow.scale import Scaling
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

import scipy as sp
import numpy as np

run_logger = logging.getLogger(__name__)


class Args:
    pass


def try_solve(instance, parameters):
    try:
        logger.setLevel(logging.WARNING)
        scaling = compute_scaling(instance, parameters)

        params = Params(time_limit=60.,
                        scaling=scaling,
                        scaling_type=ScalingType.Custom,
                        step_control_type=StepControlType.Optimizing)

        result = instance.solve(params)

        return 1 if result.success else 0

    except Exception as e:
        print(f"Error solving {instance.name}")
        print(e)
        return 0


def compute_scaling(instance, parameters):
    x0 = instance.x0

    [a, b, c] = parameters

    problem = instance.problem
    grad = problem.grad(x0)

    norm_grad = np.linalg.norm(grad, ord=1)
    normed_grad = grad / norm_grad

    log_grad = np.log(np.abs(grad) + 1)
    sqrt_log_grad = np.sqrt(np.abs(log_grad))

    value = a*normed_grad + b*log_grad + c*sqrt_log_grad

    var_weights = value.astype(np.int64)

    cons_weights = np.zeros((0,), np.int64)

    return Scaling(var_weights, cons_weights)


def solve(instances, parameters):

    num_solved = 0

    num_procs = cpu_count()

    run_logger.info("Solving %d instances", len(instances))

    with ProcessPoolExecutor(num_procs) as pool:
        futures = [
            pool.submit(try_solve, instance, parameters)
            for instance in instances
        ]

        wait(futures)

        for future in futures:
            num_solved += future.result()

    return num_solved


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    from .cutest_runner import CUTestRunner

    args = Args()
    args.name = None
    args.max_size = 100
    args.unconstrained = True
    args.verbose = False
    args.parallel = False

    runner = CUTestRunner()
    instances = runner.filter_instances(args)

    parameters = np.array([0., 0., 0.])

    def obj(parameters):
        return -solve(instances, parameters)

    res = sp.optimize.minimize(obj,
                               parameters,
                               method="Nelder-Mead")

    print("Result")
    print(res)

    print("Solution")
    print(res.x)


if __name__ == "__main__":
    main()
