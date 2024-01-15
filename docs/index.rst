.. pygradflow documentation master file, created by
   sphinx-quickstart on Sun Dec 10 23:44:36 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pygradflow's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: pygradflow
   :members:

The `pygradflow` package is designed to solve nonlinear optimization
problems given by an objective
:math:`f : \mathbb{R}^{n} \to \mathbb{R}`,
a set of smooth
nonlinear constraints
:math:`c : \mathbb{R}^{n} \to \mathbb{R}^{m}`
bounded by :math:`l, u \in \mathbb{R}^{m}`,
and variable bounds :math:`x_l, x_u \in \mathbb{R}^{n}`,
yielding the problem
     .. math::
        \begin{align}
            \min_{x \in \mathbb{R}^{n}} \quad & f(x) \\
            \text{s.t.} \quad & l \leq c(x) \leq u \\
                              & l^x \leq x \leq u^x
        \end{align}

The method works by iterating towards a primal / dual solution
approximately satisfying the
`KKT conditions <https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions>`_.

Examples
========

.. include:: examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
