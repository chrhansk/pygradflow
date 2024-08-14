import numpy as np
import scipy as sp


class BoxObjective:
    def __init__(self, iterate, lamb, rho):
        self.iterate = iterate
        self.lamb = lamb
        self.rho = rho

    def _obj_at(self, x, obj, cons):
        iterate = self.iterate
        lamb = self.lamb
        rho = self.rho

        xhat = iterate.x
        yhat = iterate.y

        dx = x - xhat
        dx_norm_sq = np.dot(dx, dx)

        w = -1 / lamb * cons
        dy = w - yhat
        dy_norm_sq = np.dot(dy, dy)

        curr_obj = obj + 0.5 * rho * np.dot(cons, cons)
        curr_obj += 0.5 * lamb * (dx_norm_sq + dy_norm_sq)

        return curr_obj

    def obj_at(self, iterate):
        x = iterate.x
        obj = iterate.obj
        cons = iterate.cons
        return self._obj_at(x, obj, cons)

    def obj(self, x):
        iterate = self.iterate
        eval = iterate.eval

        obj = eval.obj(x)
        cons = eval.cons(x)

        return self._obj_at(x, obj, cons)

    def grad(self, x):
        iterate = self.iterate
        lamb = self.lamb
        rho = self.rho

        eval = iterate.eval
        xhat = iterate.x
        yhat = iterate.y

        obj_grad = eval.obj_grad(x)
        cons = eval.cons(x)
        cons_jac = eval.cons_jac(x)

        dx = x - xhat
        cons_jac_factor = (rho + (1 / lamb)) * cons + yhat
        return obj_grad + lamb * dx + cons_jac.T.dot(cons_jac_factor)

    def hess(self, x):
        iterate = self.iterate
        lamb = self.lamb
        rho = self.rho

        eval = iterate.eval
        (n,) = x.shape

        yhat = iterate.y

        cons = eval.cons(x)
        jac = eval.cons_jac(x)
        cons_factor = 1 / lamb + rho
        y = cons_factor * cons + yhat

        hess = eval.lag_hess(x, y)
        hess += sp.sparse.diags([lamb], shape=(n, n))
        hess += cons_factor * (jac.T @ jac)

        return hess
