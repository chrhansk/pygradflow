import logging

import numpy as np

from pygradflow.solver import Solver

logging.basicConfig(level=logging.INFO)

rosenbrock = Rosenbrock()
solver = Solver(problem=rosenbrock)

solution = solver.solve()

print(solution.x)
