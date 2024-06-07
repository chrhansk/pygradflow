import numpy as np

from pygradflow.status import SolverStatus


class SolverResult:
    """
    The result of a solution of a :py:class:`pygradflow.problem.Problem`
    instance with a :py:class:`pygradflow.solver.Solver`
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        status: SolverStatus,
        iterations: int,
        num_accepted_steps: int,
        total_time: float,
        dist_factor: float,
        **attrs
    ):
        self._attrs = attrs

        self._x = x
        self._y = y
        self._d = d
        self._status = status
        self.iterations = iterations
        self.num_accepted_steps = num_accepted_steps
        self.total_time = total_time
        self.dist_factor = dist_factor

    @property
    def status(self) -> SolverStatus:
        """
        The status of the solve as a :py:class:`pygradflow.solver.SolverStatus`
        """
        return self._status

    def __getattr__(self, name):
        attrs = super().__getattribute__("_attrs")
        return attrs.get(name, None)

    def __setitem__(self, name, value):
        self._attrs[name] = value

    def __getitem__(self, name, value):
        return self._attrs[name]

    @property
    def x(self) -> np.ndarray:
        """
        The primal solution :math:`x \\in \\mathbb{R}^{n}`
        """
        return self._x

    @property
    def y(self) -> np.ndarray:
        """
        The dual solution :math:`y \\in \\mathbb{R}^{m}`
        """
        return self._y

    @property
    def d(self) -> np.ndarray:
        """
        The dual solution :math:`d \\in \\mathbb{R}^{n}`
        with respect to the variable bounds
        """
        return self._d

    def __repr__(self) -> str:
        return "SolverResult(status={0})".format(self.status)

    @property
    def success(self):
        return SolverStatus.success(self.status)
