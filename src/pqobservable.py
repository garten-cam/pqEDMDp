"""
Author: Camilo Garcia Tenorio
Returns an observable object according to the type of polynomial
the type and the p, q parameters
"""

import numpy as np
from sympy import Matrix, lambdify, symbols
from sympy.polys import orthopolys as op


class pqObservable:

    def __init__(self, p: int = 2, q: float = 0.8, l: int = 2) -> None:  # noqa: E741
        self.obs_p = p  # max polynomial order
        self.obs_q = q  # q quasi norm
        self.obs_l = l  # number of variables to observe

    def obs_fun(self) -> callable:
        x = symbols(f"x:{self.obs_l}")
        p_b = self.poly_base()
        return lambdify(x, Matrix(p_b), modules="numpy")

    def poly_base(self) -> np.ndarray:
        x = symbols(f"x:{self.obs_l}")
        orders = self.pq_mat()
        o_sym = self.assign_poly(orders, x)
        poly_b = np.prod(o_sym, axis=0)
        return poly_b

    def pq_mat(self) -> np.ndarray:
        pm = self.p_matrix()  # get the p matrix
        # reduce according to the q-quasi norm
        pqm = pm[:, np.round(np.linalg.norm(
            pm, ord=self.obs_q, axis=0), 3) <= self.obs_p]
        # sort in lexicographical ordering
        pqm = pqm[:, np.argsort(np.linalg.norm(
            pqm, axis=0, ord=self.obs_p), kind="stable")]
        return pqm

    def p_matrix(self) -> np.ndarray:
        pm = np.zeros(
            (self.obs_l, ((self.obs_p + 1) ** (self.obs_l)) - 1), dtype=int)
        if (self.obs_p ** (self.obs_l) - 1) > 9.0e18:
            raise ValueError(
                "number of state variables and p value\
                    combination exceeds maximum size"
            )
        else:
            # preallocate the final matrix
            for col in range(1, (self.obs_p + 1) ** (self.obs_l)):
                base_string = np.base_repr(col, self.obs_p + 1)
                base_string = "0" * \
                    (self.obs_l - len(base_string)) + base_string
                pm[:, col - 1] = np.flip([int(x) for x in base_string])
        return pm

    def assign_poly(self, orders, xsym) -> list:
        # Monomials, never use these, they are not orthogonal.
        return [[x**ord for ord in orders[c]] for c, x in enumerate(xsym)]

    def __eq__(self, other) -> bool:
        equal = False
        # If the shapes are not equal, they are definitely not equal
        if self.pq_mat().shape == other.pq_mat().shape:
            # If the shapes are equal then... Compare the whole array
            equal = (self.pq_mat() == other.pq_mat()).all()
        return equal

    def __hash__(self):
        return hash((self.obs_p))


class hermiteObs(pqObservable):
    def assign_poly(self, orders, xsym):
        return [
            [op.hermite_poly(ord, x) for ord in orders[c]] for c, x in enumerate(xsym)
        ]


class laguerreObs(pqObservable):
    def assign_poly(self, orders, xsym):
        return [
            [op.laguerre_poly(ord, x) for ord in orders[c]] for c, x in enumerate(xsym)
        ]


class chebyshevtObs(pqObservable):
    def assign_poly(self, orders, xsym):
        return [
            [op.chebyshevt_poly(ord, x) for ord in orders[c]]
            for c, x in enumerate(xsym)
        ]


class chebyshevuObs(pqObservable):
    def assign_poly(self, orders, xsym):
        return [
            [op.chebyshevu_poly(ord, x) for ord in orders[c]]
            for c, x in enumerate(xsym)
        ]


class legendreObs(pqObservable):
    def assign_poly(self, orders, xsym):
        return [
            [op.legendre_poly(ord, x) for ord in orders[c]] for c, x in enumerate(xsym)
        ]


if __name__ == "__main__":
    # obs = hermiteObs(l=3, p=2, q=0.9)
    obs = pqObservable(l=4, p=3, q=1.2)
    obs_function = obs.obs_fun()
    xx = obs_function(3, 5, 6, 1)
    print(xx.T)
    hobs = hermiteObs(l=4, p=4, q=1.2)
    hermite_obs = hobs.obs_fun()
    hx = hermite_obs(4, 6, 5, 6)
    print(hx.T)
    lobs = laguerreObs(l=4, p=4, q=1.2)
    lag_obs = lobs.obs_fun()
    lx = lag_obs(4, 6, 5, 6)
    print(lx.T)
    leobs = legendreObs(l=4, p=4, q=0.8)
    leg_obs = leobs.obs_fun()
    lex = lag_obs(4, 6, 5, 6)
    print(lex.T)
