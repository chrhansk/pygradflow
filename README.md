# PyGradFlow

PyGradFlow is a simple implementation of the sequential homotopy method to
be used to solve general nonlinear programs.

To solve a nonlinear problem, subclass the `Problem` class, implementing
its abstract methods, pass a problem instance to a `Solver`, and call
its `solve()` method.

Note: This code is for research purposes, not productive use.


## References

- Potschka, A., Bock, H.G. A sequential homotopy method for mathematical programming problems. Math. Program. 187, 459–486 (2021). https://doi.org/10.1007/s10107-020-01488-z
- Pearson, John W., and Andreas Potschka. "A preconditioned inexact active-set method for large-scale nonlinear optimal control problems." arXiv preprint arXiv:2112.05020 (2021).
