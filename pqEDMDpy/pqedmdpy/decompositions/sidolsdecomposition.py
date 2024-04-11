"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on subspace identification.
The solution is ols.
"""

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pqedmdpy.decompositions import siddecomposition
import matplotlib.pyplot as plt


class sidOlsDecomposition(siddecomposition.sidDecomposition):
    def __init__(
        self,
        observable,
        system,
    ):
        self.observable = observable
        xeval, u = self.xu_eval(system)
        if "u" in system[0].keys():
            # if there is an input to the system
            self.m = np.shape(system[0]["u"])[1]
        else:
            self.m = 0

        self.n_obs = np.shape(self.observable.pq_mat())[1]
        self.l = self.observable.l
        # calculates a sensible number of blocks
        self.hankel_blocks(xeval)
        # Total number of samples
        tot_sam = np.sum([sys_i["y"].shape[0] for sys_i in system])

        Ysid = np.hstack(
            [self.block_hankel(per_x.T, 2 * self.hl_bl) for per_x in xeval]
        ) / tot_sam
        Usid = np.hstack([self.block_hankel(per_u.T, 2 * self.hl_bl)
                         for per_u in u]) / tot_sam
        # Compute Gamma
        self.computeGamma(Ysid, Usid)
        # This is where the algorithms diverge.
        # I need Oi from the last step, but it is better to recompute it
        self.AC(Ysid, Usid)
        self.recomputeGamma()
        self.BD(xeval, u)
        self.edmdC = self.matrix_C()[:, 1:]

    def AC(self, Ysid, Usid):
        # I need the same gamma minus matrices
        gmm = self.Gamma[: -self.n_obs, :]
        # Inverses of the gammas
        gam_inv = np.linalg.pinv(self.Gamma)
        gmm_inv = np.linalg.pinv(gmm)
        # I also need the Yf, Uf, and Wp matrices
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

        # Get oblique projections
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
        sol = self.svd_solution(lhs.T, rhs.T).T
        self.A = sol[:self.n, :self.n]
        # self.B = sol[:self.n, self.n:]
        self.C = sol[self.n:, :self.n]
        # self.D = sol[self.n:, self.n:]
