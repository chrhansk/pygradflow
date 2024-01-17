import os

import pyqplib

from pygradflow.solver import Problem, Solver

from .instance import Instance
from .runner import Runner


class QPLibProblem(Problem):
    def __init__(self, problem):
        self.problem = problem

        super().__init__(problem.var_lb,
                         problem.var_ub,
                         cons_lb=problem.cons_lb,
                         cons_ub=problem.cons_ub)

    def obj(self, x):
        return self.problem.obj_val(x)

    def obj_grad(self, x):
        return self.problem.obj_grad(x)

    def cons(self, x):
        return self.problem.cons_val(x)

    def cons_jac(self, x):
        return self.problem.cons_jac(x)

    def lag_hess(self, x, y):
        return self.problem.lag_hess(x, y)

    @property
    def x0(self):
        return self.problem.x0

    @property
    def y0(self):
        return self.problem.y0


class QPLibInstance(Instance):
    def __init__(self, description):
        self.description = description

        super().__init__(description.name,
                         description.num_vars,
                         description.num_cons)

    @property
    def filename(self):
        return self.description.filename


class QPLibRunner(Runner):

    def __init__(self):
        super().__init__(name="qplib")

    def solve_instance(self, instance, params):
        qproblem = pyqplib.read_problem(instance.filename)
        problem = QPLibProblem(qproblem)
        solver = Solver(problem, params)
        return solver.solve(problem.x0, problem.y0)

    def get_instances(self, args):
        instances = []
        for file in os.listdir(args.directory):
            is_qplib = file.endswith(".qplib") or file.endswith(".qplib.zip")

            if is_qplib:
                filename = os.path.join(args.directory, file)
                description = pyqplib.read_description(filename)
                instances.append(QPLibInstance(description))

        return instances

    def parser(self):
        parser = super().parser()
        parser.add_argument('directory')
        return parser


if __name__ == "__main__":
    QPLibRunner().main()
