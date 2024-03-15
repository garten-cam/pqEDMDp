"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on subspace identification
"""

import numpy as np

from pqedmdpy.decompositions import pqdecomposition


class sidDecomposition(pqdecomposition.pqDecomposition):
    def __init__(
        self,
        observable,
        system,
    ):
        # Class to perform the regression
        # and use the SVD as the inverse engine
        self.observable = observable
        # Evaluate the data
        xeval, u = self.xu_eval(system)
        if "u" in system[0].keys():
            # if there is an input to the system
            self.m = np.shape(system[0]["u"])[1]
        else:
            self.m = 0

        self.n_obs = np.shape(self.observable.pq_mat())[1]
        # Samples per trajectory get the lenght of all the trajectories in the
        # training set
        # I only need the minimum to set the hankel blocks
        min_spl = np.min([sp['y'].shape[0] for sp in system])
        # maximize the number of blocks while keeping the hankel matrix fat
        self.hl_bl = int(np.floor(min_spl / (self.n_obs * len(system))))
        Ysid = np.hstack(
            [self.block_hankel(per_x.T, 2 * self.hl_bl) for per_x in xeval]
        )
        Usid = np.hstack(
            [self.block_hankel(per_u.T, 2 * self.hl_bl) for per_u in u]
        )
        # Perform the oblique projection
        Oi = self.oblique_prj(Usid, Ysid)
        # and the orthogonal projection to make it a MOESP
        WOW = self.z_proj_O_x(Oi, Usid[self.hl_bl * self.m:, :])
        # Finally, the singular value decomposition
        U, S, _ = np.linalg.svd(WOW)
        self.order = int(np.sum(S / np.max(S) > 0.1))  # This might be up for debate
        # Trim the matrices according to the order
        # r
        U = U[:, :self.order]
        S = S[:self.order]
        # Calculate the Gamma matrix
        self.Gamma = U @ np.diag(np.sqrt(S))
        # In python, I can modify the object in any method.
        # I do not need to return the matrices.
        res = self.linear_sol_ABCD(Ysid, Usid)
        self.Kalman_gain(res)

    def linear_sol_ABCD(self, Ysid, Usid):
        # # Needs refactoring... too long and unreadable... # #
        # From the calculated Gamma, and the hankel matrices,
        # Get the A, B, C and D matrices
        # The Gamma minus mat
        gmm = self.Gamma[:-self.n_obs, :]
        # Inverses of the gammas
        gam_inv = np.linalg.pinv(self.Gamma)
        gmm_inv = np.linalg.pinv(gmm)

        # Linear equations for A, C
        # z_i = Yf/[Wp;Uf]
        wp = np.vstack((
            Usid[:self.hl_bl * self.m, :],  # U_past
            Ysid[:self.hl_bl * self.n_obs, :]  # Y_past
        ))
        z_i = self.z_proj_x(
            Ysid[self.hl_bl * self.n_obs:, :],  # Y_future
            np.vstack((
                wp,
                Usid[self.hl_bl * self.m:, :]  # U_future
            ))
        )
        # zip = Yf-/[Wp+;Uf-]
        wpp = np.vstack((
            Usid[:(self.hl_bl + 1) * self.m, :],  # U_past+
            Ysid[:(self.hl_bl + 1) * self.n_obs, :]  # Y_past+
        ))
        zip = self.z_proj_x(
            Ysid[(self.hl_bl + 1) * self.n_obs:, :],
            np.vstack((
                wpp,
                Usid[(self.hl_bl + 1) * self.m:, :]  # U_future+
            ))
        )
        # left-hand-side = (gmm_inv*zip;Yii)
        lhs = np.vstack((
            gmm_inv @ zip,
            Ysid[self.hl_bl * self.n_obs:(self.hl_bl + 1) * self.n_obs, :]
        ))
        # right-hand-side = (gam_inv*z_i;Uf)
        rhs = np.vstack((
            gam_inv @ z_i,
            Usid[self.hl_bl * self.m:, :]  # U_future
        ))
        AC = lhs @ np.linalg.pinv(rhs)
        # I do not need to compute and then extract.
        self.A = AC[:self.order, :self.order]
        self.C = AC[self.order:, :self.order]
        # Calculate the residuals
        res = lhs - AC @ rhs

        # From A and C, recompute Gamma
        self.recompute_Gamma()
        # And get the Gamma- and the inverses again
        gmm = self.Gamma[:-self.n_obs, :]
        # Inverses of the gammas
        gam_inv = np.linalg.pinv(self.Gamma)
        gmm_inv = np.linalg.pinv(gmm)

        # Here I am following the alg from the book. I do not understand it 
        # completely yet. pg: 125

        P = lhs - np.vstack((self.A, self.C)) @ rhs[:self.order, :]
        Q = Usid[self.hl_bl * self.m:, :]  # U_future

        L1 = self.A @ gam_inv
        L2 = self.C @ gam_inv

        M = np.hstack((np.zeros((self.order, self.n_obs)), gmm_inv))
        X = np.vstack((
            np.hstack((np.eye(self.n_obs), np.zeros((self.n_obs, self.order)))),
            np.hstack((np.zeros((self.n_obs * (self.hl_bl - 1), self.n_obs)), gmm))
        ))

        totm = 0
        for k in range(self.hl_bl):
            N = np.vstack((
                np.hstack((M[:, k * self.n_obs:] - L1[:, k * self.n_obs:],
                           np.zeros((self.order, k * self.n_obs)))),
                np.hstack((-L2[:, k * self.n_obs:],
                           np.zeros((self.n_obs, k * self.n_obs))))
            ))
            if k == 0:
                N[self.order:, :self.n_obs] = np.eye(self.n_obs) + N[self.order:, :self.n_obs]

            N = N @ X
            totm += np.kron(Q[k * self.m:(k + 1) * self.m, :].T, N)

        bd_vec = np.linalg.lstsq(totm, P.reshape((-1, 1), order='F'))[0]
        bd = bd_vec.reshape((self.order + self.n_obs, self.m), order='F')
        # Different from matlab, I can mofify the object here
        self.B = bd[:self.n_obs, :]
        self.D = bd[self.n_obs:self.n_obs + self.order, :]
        return res

    def Kalman_gain(self, res):
        return res

    def recompute_Gamma(self):
        # Asseign the new C to the Gamma matrix
        self.Gamma[:self.n_obs, :] = self.C
        for blk in range(1, self.hl_bl):
            self.Gamma[
                blk * self.n_obs:(blk + 1) * self.n_obs, :
            ] = self.Gamma[
                (blk - 1) * self.n_obs:blk * self.n_obs
            ] @ self.A

    def oblique_prj(self, Usid, Ysid):
        # Oi = Yf/_{Uf}(Wp) the oblique projection of Y future, onto
        # the row space of Wp along U future. For that,
        # Oi = Yf/_{Uf}(Wp) = (Yf/oUf)*pinv(Wp/oUf)*Wp
            
        # Yf/oUf Y_future projected onto the otrthogonal complement of U_future
        yfPOuf = self.z_proj_O_x(
            Ysid[self.hl_bl * self.n_obs:, :],  # Y_future
            Usid[self.hl_bl * self.m:, :]  # U_future
        )
        # Wp/oUf Wp=[Up;Yp] The instrumental variable projected onto the
        # orthogonal complement of U_future
        wp = np.vstack((
            Usid[:self.hl_bl * self.m, :],  # U_past
            Ysid[:self.hl_bl * self.n_obs, :]  # Y_past
        ))
        wpPOuf = np.linalg.pinv(
            self.z_proj_O_x(
                wp,
                Usid[self.hl_bl * self.m:, :]
            )
        )        # return the oblique projection
        return yfPOuf @ wpPOuf @ wp

    def xu_eval(self, system):
        # for sid, just evaluate the outputs
        obsrv = self.observable.obs_fun()
        ytr = [np.squeeze(obsrv(*sp["y"].T).T) for sp in system]
        if "u" in system[0].keys():
            utr = [sp["u"] for sp in system]
        else:
            utr = []

        return ytr, utr

    @staticmethod
    def block_hankel(mat, s):
        # mat is the matrix to hankelize,
        # s is the number of blocks
        j_samples = mat.shape[1]
        # number of columns in the final matrix
        n_col = int(j_samples - 2 * s + 1)
        # Assign the values
        return np.vstack([mat[:, blk:blk + n_col] for blk in range(s)]) # This can be a one liner

    @staticmethod
    def z_proj_O_x(z, x):
        # z projected onto the row space of  the orthogonal complement of x
        p = z.shape[0]
        Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        R = R.T
        return R[-p:, -p:] @ Q[:, -p:].T

    @staticmethod
    def z_proj_x(z, x):
        # z projected onto the row space of x
        p = z.shape[0]
        Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        R = R.T
        return R[-p:, :-p] @ Q[:, :-p].T
