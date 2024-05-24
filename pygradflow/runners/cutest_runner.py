from functools import cached_property

import numpy as np
import pycutest

from pygradflow.problem import Problem

from .instance import Instance
from .runner import Runner


def cutest_is_ne_prob(name):
    return name.endswith("NE")


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
        assert instance.m == 0
        super().__init__(var_lb, var_ub, num_cons=0)

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

    def x0(self):
        return self.instance.x0


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

    def cons_jac(self, x):
        cons, jac = self.instance.scons(x, gradient=True)
        return jac

    def lag_hess(self, x, y):
        return self.instance.sphess(x, v=y)

    @property
    def x0(self):
        return self.instance.x0()


# Nonlinear equations: Goal is to minimize the violation
# 1/2 ||c(x)||^2 subject to problem bounds. Constraint
# functions are used to access the residuals and their derivatives
class NECUTEstProblem(Problem):
    def __init__(self, instance):
        self.instance = instance
        var_lb = cutest_map_inf(instance.bl)
        var_ub = cutest_map_inf(instance.bu)

        super().__init__(var_lb, var_ub, num_cons=0)

    @cached_property
    def problem(self):
        return pycutest.import_problem(self.name, drop_fixed_variables=True)

    def obj(self, x):
        residuals = self.instance.cons(x)
        return 0.5 * np.dot(residuals, residuals)

    def obj_grad(self, x):
        residuals, jac = self.instance.scons(x, gradient=True)
        return jac.T.dot(residuals)

    def lag_hess(self, x, y):
        _, jac = self.instance.scons(x, gradient=True)
        return jac.T.dot(jac)

    @property
    def x0(self):
        return self.instance.x0

    @property
    def y0(self):
        return np.zeros((self.num_cons,))


class CUTestInstance(Instance):
    def __init__(self, instance):
        self.instance = instance

        props = pycutest.problem_properties(self.instance)

        num_vars = props["n"]
        num_cons = props["m"]

        if num_cons is None:
            num_cons = 0

        super().__init__(instance, num_vars, num_cons)

    def cutest_problem(self):
        return pycutest.import_problem(self.name, drop_fixed_variables=True)

    def x0(self):
        cutest_problem = self.cutest_problem()
        return cutest_problem.x0

    def problem(self):
        cutest_problem = self.cutest_problem()

        is_ne = cutest_is_ne_prob(self.name)

        if is_ne:
            return NECUTEstProblem(cutest_problem)
        elif cutest_problem.m == 0:
            return UnconstrainedCUTEstProblem(cutest_problem)
        else:
            return ConstrainedCUTEstProblem(cutest_problem)


class CUTestRunner(Runner):
    def __init__(self):
        super().__init__(name="cutest")

    def parser(self):
        parser = super().parser()

        parser.add_argument("--ignore_ne_probs", action="store_true")

        return parser

    def get_instances(self, args):
        instances = pycutest.find_problems()
        filtered_instances = []

        for instance in instances:
            props = pycutest.problem_properties(instance)

            if args.ignore_ne_probs and cutest_is_ne_prob(instance):
                continue

            if props["m"] == "variable":
                continue

            if props["n"] == "variable":
                continue

            filtered_instances.append(CUTestInstance(instance))

        return filtered_instances


if __name__ == "__main__":
    CUTestRunner().main()
