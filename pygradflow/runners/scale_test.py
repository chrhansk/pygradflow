from abc import ABC, abstractmethod

import logging

from pygradflow.log import logger
from pygradflow.params import Params, ScalingType, StepControlType
from pygradflow.scale import Scaling
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

import scipy as sp
import numpy as np

run_logger = logging.getLogger(__name__)


class ScalingRule(ABC):
    @abstractmethod
    def compute_scaling(self, instance):
        raise NotImplementedError()


class ZeroScaling(ScalingRule):
    def compute_scaling(self, instance):

        return Scaling.zero(instance.num_vars,
                            instance.num_cons)


class SimpleGrad(ScalingRule):
    def compute_scaling(self, instance):

        x0 = instance.x0

        problem = instance.problem
        grad = problem.grad(x0)

        grad_weights = Scaling.weights_from_nominal_values(np.abs(grad))
        var_weights = -grad_weights

        cons_weights = np.zeros((0,), np.int64)

        return Scaling(var_weights, cons_weights)


class Args:
    pass


def try_solve(instance, scaling_rule):
    try:
        logger.setLevel(logging.WARNING)

        scaling = scaling_rule.compute_scaling(instance)

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


def solve(instances, scaling_rule):

    num_solved = 0

    num_procs = cpu_count()

    run_logger.info("Solving %d instances", len(instances))

    with ProcessPoolExecutor(num_procs) as pool:
        futures = [
            pool.submit(try_solve, instance, scaling_rule)
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

    scaling_rules = [ZeroScaling(), SimpleGrad()]

    args = Args()
    args.name = None
    args.max_size = 100
    args.unconstrained = True
    args.verbose = False
    args.parallel = False

    runner = CUTestRunner()
    instances = runner.filter_instances(args)

    scaling_num_solved = []

    for scaling_rule in scaling_rules:
        num_solved = solve(instances, scaling_rule)
        scaling_num_solved.append(num_solved)

    for (scaling_rule, num_solved) in zip(scaling_rules, scaling_num_solved):
        print(f"{scaling_rule.__class__.__name__}: {num_solved} instances solved")


if __name__ == "__main__":
    main()
