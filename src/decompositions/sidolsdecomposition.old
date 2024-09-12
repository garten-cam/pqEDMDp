"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on subspace identification.
The solution is ols.
"""

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pqedmdpy.pqObservable import pqObservable
from pqedmdpy.decompositions import siddecomposition
import matplotlib.pyplot as plt


class sidOlsDecomposition(siddecomposition.sidDecomposition):
    def __init__(
        self,
        observable: pqObservable,
        system: dict,
    ):
        self.observable = observable
        xeval, u = self.xu_eval(system)
        if "u" in system[0].keys():
            # if there is an input to the system
            n_in = np.shape(system[0]["u"])[1]
        else:
            n_in = 0

        self.m: int = n_in
        self.n_obs: int = np.shape(self.observable.pq_mat())[1]
        self.l: int = self.observable.l
        # calculates a sensible number of blocks
        self.hankel_blocks(xeval)
        # Total number of samples

        Ysid = np.hstack(
            [self.block_hankel(per_x.T, 2 * self.hl_bl) for per_x in xeval]
        )
        if "u" in system[0].keys():
            # if there is an input to the system
            Usid = np.hstack([self.block_hankel(per_u.T, 2 * self.hl_bl)
                              for per_u in u])
        else:
            Usid = np.empty((2 * self.hl_bl * self.m, Ysid.shape[1]))
        # Compute Gamma
        # self.computeGamma(Ysid, Usid)
        # This is where the algorithms diverge.
        # I need Oi from the last step, but it is better to recompute it
        self.ACKn(Ysid, Usid)
        self.recomputeGamma()
        self.BD(xeval, u)
        self.edmdC = self.matrix_C()[:, 1:]

    def n_cost(self, n, U, S, Ysid, Usid):
        Un = U[:, : n]
        Sn = S[: n]
        # Gamma, gamma_minus and their inverses
        gam = Un @ np.diag(np.sqrt(Sn))
        gmm = gam[: -self.n_obs, :]
        gam_inv = np.linalg.pinv(gam)
        gmm_inv = np.linalg.pinv(gmm)
        Yf = Ysid[self.hl_bl * self.n_obs:, :]
        Uf = Usid[self.hl_bl * self.m:, :]
        Wp = np.vstack(
            (
                Usid[: self.hl_bl * self.m, :],  # U_past
                Ysid[: self.hl_bl * self.n_obs, :],  # Y_past
            )
        )
        # Also, the Yf-, Uf-, and Wp+
        Yfm = Ysid[(self.hl_bl + 1) * self.n_obs:, :]
        Ufm = Usid[(self.hl_bl + 1) * self.m:, :]
        Wpp = np.vstack(
            (
                Usid[: (self.hl_bl + 1) * self.m, :],  # U_past
                Ysid[: (self.hl_bl + 1) * self.n_obs, :],  # Y_past
            )
        )
        Oi = self.oblique_prj(Yf, Uf, Wp)
        Oip = self.oblique_prj(Yfm, Ufm, Wpp)
        # And the WOWs
        WOiW = self.z_proj_O_x(Oi, Uf)
        WOipW = self.z_proj_O_x(Oip, Ufm)
        # Get the sequence of states
        Xi = gam_inv @ WOiW
        Xip = gmm_inv @ WOipW
        # Calculate the solution
        lhs = np.vstack((
            Xi,
            Ysid[(self.hl_bl - 1) *
                 self.n_obs: (self.hl_bl) * self.n_obs, :],
        ))
        rhs = np.vstack((
            Xip,
            Usid[(self.hl_bl - 1) * self.m:self.hl_bl * self.m]
        ))
        ac = self.svd_solution(lhs.T, rhs.T).T
        a = ac[:n, :n]
        # self.B = sol[:self.n, self.n:]
        c = ac[n:, :n]
        res = lhs - ac @ rhs
        cost = np.sum(np.abs(res))
        return cost, res, a, c
