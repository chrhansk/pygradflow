from pygradflow.problem import Problem


class QP(Problem):
    def __init__(self, hess, jac, grad, res, lb, ub):
        self.m, self.n = jac.shape
        assert hess.shape[0] == self.n
        assert hess.shape[1] == self.n
        assert grad.shape[0] == self.n
        assert res.shape[0] == self.m
        super().__init__(lb, ub, num_cons=self.m)
        self.A = hess
        self.B = jac
        self.a = grad
        self.b = res

    def obj(self, x):
        return 0.5 * x.T @ self.A @ x + self.a.T @ x

    def obj_grad(self, x):
        return self.A @ x + self.a

    def cons(self, x):
        return self.B @ x + self.b

    def cons_jac(self, x):
        return self.B

    def lag_hess(self, x, _):
        return self.A
