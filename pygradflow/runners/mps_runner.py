import os

import mpspy
import numpy as np

from pygradflow.problem import Problem
from pygradflow.util import sparse_zero

from .instance import Instance
from .runner import Runner


class LinearProblem(Problem):
    def __init__(self, instance):
        self.instance = instance

        super().__init__(
            var_lb=instance.var_lb,
            var_ub=instance.var_ub,
            cons_lb=instance.cons_lb,
            cons_ub=instance.cons_ub,
        )

    def obj(self, x):
        instance = self.instance
        return np.dot(instance.obj, x)

    def obj_grad(self, x):
        instance = self.instance
        return instance.obj

    def cons(self, x):
        instance = self.instance
        coeffs = instance.coeffs
        return coeffs.dot(x)

    def cons_jac(self, x):
        instance = self.instance
        return instance.coeffs

    def lag_hess(self, x, y):
        num_vars = self.num_vars
        return sparse_zero(shape=(num_vars, num_vars))


class MPSInstance(Instance):
    def __init__(self, instance):
        self.instance = instance

        num_vars = instance.num_vars
        num_cons = instance.num_cons
        name = instance.name

        super().__init__(name, num_vars, num_cons)

    def problem(self):
        return LinearProblem(self.instance)

    def x0(self):
        instance = self.instance
        num_vars = instance.num_vars

        x0 = np.zeros((num_vars,))

        return np.clip(x0, instance.var_lb, instance.var_ub)


class MPSRunner(Runner):
    def __init__(self):
        super().__init__(name="mps")

    def get_instances(self, args):
        instances = []
        for file in os.listdir(args.directory):
            is_qplib = file.endswith(".mps") or file.endswith(".mps.gz")

            if is_qplib:
                filename = os.path.join(args.directory, file)
                description = mpspy.read_mps(filename)
                instances.append(MPSInstance(description))

        return instances

    def parser(self):
        parser = super().parser()
        parser.add_argument("directory")
        return parser


if __name__ == "__main__":
    MPSRunner().main()
