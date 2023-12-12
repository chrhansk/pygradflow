Consider the problem of minimizing the Rosenbrock function

 .. math::

    f(x_0, x_1) := (a - x_0)^{2} + b(x_1 - x_0^{2})^{2}

over :math:`\mathbb{R}^{2}` with parameters :math:`(a, b) = (1, 100)`.
This problem in unconstrained
in the sense that it incorporates neither variable bounds
not nonlinear constraints. To solve the problem, we
first create a subclass of the :py:class:`pygradflow.problem.Problem`
class, overriding the required methods to evaluate :math:`f`,
:math:`\nabla f`, and
:math:`\nabla_{xx} \mathcal{L}(x, y) = \nabla_{xx} f(x)`:

.. literalinclude :: rosenbrock.py
   :language: python

To solve the problem, we create a :py:class:`pygradflow.solver.Solver`
based on the problem definition and call its :py:func:`pygradflow.solver.Solver.solve`
method:

.. literalinclude :: solve_rosenbrock.py
   :language: python

The resulting :py:class:`pygradflow.solver.SolverResult` contains a status, indicating
whether or not the solution attempt was successful, along with the primal
solution corresponding, which in this case is close to the known optimum
of :math:`(x_0^{*}, x_1^{*}) = (1, 1)`:

.. literalinclude :: solve_rosenbrock.output
   :language: text
