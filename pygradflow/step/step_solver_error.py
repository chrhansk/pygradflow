class StepSolverError(Exception):
    """
    Error signaling that the step solver failed, e.g. because the
    Newton matrix is (near) singular.
    """

    pass
