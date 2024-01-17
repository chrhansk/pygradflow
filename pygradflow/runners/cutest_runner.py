import logging

import numpy as np
import pycutest

from pygradflow.solver import Problem, Solver

from .instance import Instance
from .runner import Runner

formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")


def cutest_map_inf(x):
    cutest_inf = 1e20
    y = np.copy(x)
    y[y == cutest_inf] = np.inf
    y[y == -cutest_inf] = -np.inf
    return y


class UnconstrainedCUTEstProblem(Problem):
    def __init__(self, instance):
        self.instance = instance
        var_lb = cutest_map_inf(instance.bl)
        var_ub = cutest_map_inf(instance.bu)
        super().__init__(var_lb, var_ub, num_cons=instance.m)

    def obj(self, x):
        return self.instance.obj(x)

    def obj_grad(self, x):
        obj, grad = self.instance.obj(x, gradient=True)
        return grad

    def cons(self, x):
        raise NotImplementedError("Unconstrained problem")

    def cons_jac(self, x):
        raise NotImplementedError("Unconstrained problem")

    def lag_hess(self, x, y):
        return self.instance.sphess(x, v=None)

    @property
    def x0(self):
        return self.instance.x0

    @property
    def y0(self):
        return np.zeros((self.num_cons,))


class ConstrainedCUTEstProblem(Problem):
    def __init__(self, instance):
        self.instance = instance
        var_lb = cutest_map_inf(instance.bl)
        var_ub = cutest_map_inf(instance.bu)
        cons_lb = cutest_map_inf(instance.cl)
        cons_ub = cutest_map_inf(instance.cu)

        super().__init__(var_lb, var_ub, cons_lb=cons_lb, cons_ub=cons_ub)

    def obj(self, x):
        return self.instance.obj(x)

    def obj_grad(self, x):
        obj, grad = self.instance.obj(x, gradient=True)
        return grad

    def cons(self, x):
        return self.instance.cons(x)
        raise NotImplementedError("Unconstrained problem")

    def cons_jac(self, x):
        cons, jac = self.instance.scons(x, gradient=True)
        return jac

    def lag_hess(self, x, y):
        return self.instance.sphess(x, v=y)

    @property
    def x0(self):
        return self.instance.x0

    @property
    def y0(self):
        return self.instance.v0


class CUTestInstance(Instance):
    def __init__(self, instance):
        self.instance = instance

        props = pycutest.problem_properties(instance)

        num_vars = props["n"]
        num_cons = props["m"]

        if num_cons is None:
            num_cons = 0

        super().__init__(instance, num_vars, num_cons)

    def solve(self, params):
        problem = pycutest.import_problem(self.name, drop_fixed_variables=True)

        if problem.m == 0:
            problem = UnconstrainedCUTEstProblem(problem)
        else:
            problem = ConstrainedCUTEstProblem(problem)

        solver = Solver(problem, params)
        return solver.solve(problem.x0, problem.y0)


class CUTestRunner(Runner):
    def __init__(self):
        super().__init__(name="cutest")

    def get_instances(self, args):
        instances = pycutest.find_problems()
        filtered_instances = []

        for instance in instances:
            props = pycutest.problem_properties(instance)

            if props["m"] == "variable":
                continue

            if props["n"] == "variable":
                continue

            filtered_instances.append(CUTestInstance(instance))

        return filtered_instances


if __name__ == "__main__":
    CUTestRunner().main()
