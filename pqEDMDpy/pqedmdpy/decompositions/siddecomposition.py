"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on subspace identification
"""

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pqedmdpy.decompositions import svddecomposition


class sidDecomposition(svddecomposition.svdDecomposition):
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
        self.l = self.observable.l
        self.hankel_blocks(xeval)

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
        # # # self.computeGamma(Ysid, Usid)
        # In python, I can modify the object in any method.
        # I do not need to return the matrices.
        # # # self.ACK(Ysid, Usid)
        self.ACKn(Ysid, Usid)
        # From the new A and C, recompute Gamma
        self.recomputeGamma()
        # Now, from the A, C, and the data from the system, get the B and D
        # matrices.
        self.BD(xeval, u)
        self.edmdC = self.matrix_C()[:, 1:]

    def compute_svd(self, Ysid, Usid):
        # Perform the oblique projection
        # The last approach is not extendable. Refactoring oblique P
        Yf = Ysid[self.hl_bl * self.n_obs:, :]
        Uf = Usid[self.hl_bl * self.m:, :]
        Wp = np.vstack(
            (
                Usid[: self.hl_bl * self.m, :],  # U_past
                Ysid[: self.hl_bl * self.n_obs, :],  # Y_past
            )
        )
        Oi = self.oblique_prj(Yf, Uf, Wp)
        WOW = self.z_proj_O_x(Oi, Uf)
        # Finally, the singular value decomposition
        U, S, _ = np.linalg.svd(WOW)
        return U, S

    def ACKn(self, Ysid, Usid):
        U, S = self.compute_svd(Ysid, Usid)
        # possible n's from the order of the system,
        # untill the criterion from some paper
        ns = np.arange(
            self.l,
            2 * np.sum(np.log(S) > 0.5 * (np.log(S[0] + np.log(S[-1])))), 1
        )
        nv = [self.n_cost(ni, U, S, Ysid, Usid)[0] for ni in ns]
        # order of the system
        self.n = np.argmin(nv) + self.l
        # Calculate everithing
        _, res, self.A, self.C = self.n_cost(self.n, U, S, Ysid, Usid)
        # over all possible values of n, calculate the error
        # Optimize the solution for the order of the System

    def n_cost(self, n, U, S, Ysid, Usid):
        # Truncate according to the order
        Un = U[:, : n]
        Sn = S[: n]
        # Gamma, gamma_minus and their inverses
        gam = Un @ np.diag(np.sqrt(Sn))
        gmm = gam[: -self.n_obs, :]
        gam_inv = np.linalg.pinv(gam)
        gmm_inv = np.linalg.pinv(gmm)

        wp = np.vstack(
            (
                Usid[: self.hl_bl * self.m, :],  # U_past
                Ysid[: self.hl_bl * self.n_obs, :],  # Y_past
            )
        )
        z_i = self.z_proj_x(
            Ysid[self.hl_bl * self.n_obs:, :],  # Y_future
            np.vstack(
                (
                    wp,
                    Usid[self.hl_bl * self.m:, :],  # U_future
                )
            ),
        )
        # zip = Yf-/[Wp+;Uf-]
        wpp = np.vstack(
            (
                Usid[: (self.hl_bl + 1) * self.m, :],  # U_past+
                Ysid[: (self.hl_bl + 1) * self.n_obs, :],  # Y_past+
            )
        )
        zip = self.z_proj_x(
            Ysid[(self.hl_bl + 1) * self.n_obs:, :],
            np.vstack(
                (
                    wpp,
                    Usid[(self.hl_bl + 1) * self.m:, :],  # U_future+
                )
            ),
        )
        # left-hand-side = (gmm_inv*zip;Yii)
        lhs = np.vstack(
            (
                gmm_inv @ zip,
                Ysid[(self.hl_bl - 1) * self.n_obs: (self.hl_bl) * self.n_obs, :],
            )
        )
        # right-hand-side = (gam_inv*z_i;Uf)
        rhs = np.vstack(
            (
                gam_inv @ z_i,
                Usid[self.hl_bl * self.m:, :],  # U_future
            )
        )
        # ac = lhs @ np.linalg.pinv(rhs)
        ac = self.svd_solution(lhs.T, rhs.T).T
        # I do not need to compute and then extract.
        a = ac[: n, : n]
        c = ac[n:, : n]
        # Calculate the residuals
        res = lhs - ac @ rhs
        cost = np.sum(np.abs(res))
        return cost, res, a, c

    def ACK(self, Ysid, Usid):
        # Get the A, C, and Kalman gains matrices
        # The Gamma minus mat
        gmm = self.Gamma[: -self.n_obs, :]
        # Inverses of the gammas
        gam_inv = np.linalg.pinv(self.Gamma)
        gmm_inv = np.linalg.pinv(gmm)

        # Linear equations for A, C
        # z_i = Yf/[Wp;Uf]
        wp = np.vstack(
            (
                Usid[: self.hl_bl * self.m, :],  # U_past
                Ysid[: self.hl_bl * self.n_obs, :],  # Y_past
            )
        )
        z_i = self.z_proj_x(
            Ysid[self.hl_bl * self.n_obs:, :],  # Y_future
            np.vstack(
                (
                    wp,
                    Usid[self.hl_bl * self.m:, :],  # U_future
                )
            ),
        )
        # zip = Yf-/[Wp+;Uf-]
        wpp = np.vstack(
            (
                Usid[: (self.hl_bl + 1) * self.m, :],  # U_past+
                Ysid[: (self.hl_bl + 1) * self.n_obs, :],  # Y_past+
            )
        )
        zip = self.z_proj_x(
            Ysid[(self.hl_bl + 1) * self.n_obs:, :],
            np.vstack(
                (
                    wpp,
                    Usid[(self.hl_bl + 1) * self.m:, :],  # U_future+
                )
            ),
        )
        # left-hand-side = (gmm_inv*zip;Yii)
        lhs = np.vstack(
            (
                gmm_inv @ zip,
                Ysid[(self.hl_bl - 1) * self.n_obs: (self.hl_bl) * self.n_obs, :],
            )
        )
        # right-hand-side = (gam_inv*z_i;Uf)
        rhs = np.vstack(
            (
                gam_inv @ z_i,
                Usid[self.hl_bl * self.m:, :],  # U_future
            )
        )
        # ac = lhs @ np.linalg.pinv(rhs)
        ac = self.svd_solution(lhs.T, rhs.T).T
        # I do not need to compute and then extract.
        self.A = ac[: self.n, : self.n]
        self.C = ac[self.n:, : self.n]
        # Calculate the residuals
        res = lhs - ac @ rhs
        sqres = res @ res.T / (np.shape(res)[1] - 1)
        QQ = sqres[: self.n, : self.n]
        RR = sqres[self.n: self.n + self.n_obs, self.n: self.n + self.n_obs]
        SS = sqres[: self.n, self.n: self.n + self.n_obs]
        X = dare(self.A.T, self.C.T, QQ, RR, None, SS)
        # # The Kalman gain
        self.K = np.linalg.inv(self.C @ X @ self.C.T + RR) @ (
            self.C @ X @ self.A.T + SS.T
        )

    def BD(self, yeval, u):
        # I need the maximun samples in all the trajectories
        max_sam = np.max([np.shape(y_i)[0] for y_i in yeval])
        ac = [self.C for _ in range(max_sam)]
        for ac_blk in range(1, max_sam):
            ac[ac_blk] = ac[ac_blk - 1] @ self.A

        ukrac = [
            [np.kron(np.zeros((1, u_i.shape[1])), self.C)
             for _ in range(u_i.shape[0])]
            for u_i in u
        ]
        for u_in, u_i in enumerate(u):
            for u_blk in range(u_i.shape[0]):
                krcb = np.zeros((*ukrac[u_in][u_blk].shape, u_blk + 1))
                for step in range(u_blk + 1):
                    krcb[:, :, step] = np.kron(
                        u[u_in][step, :], ac[u_blk - step])
                ukrac[u_in][u_blk] = np.sum(krcb, axis=2)
        # Ok, kroneckers done... jumadre
        # Now I need the right-hand-side and the lhs
        rhsblk = [
            np.zeros((
                y_i[1:, :].size, self.n * (len(yeval) + self.m)
            )) for y_i in yeval
        ]
        lhsblk = [y_i[1:, :].T.reshape((-1, 1), order='F') for y_i in yeval]
        # Populate the rhsblk
        for b_i, _ in enumerate(rhsblk):
            rhsblk[b_i][:, : self.n * self.m] = np.vstack((ukrac[b_i][:-1]))
            rhsblk[b_i][:, self.n * (b_i + self.m): self.n *
                        (b_i + self.m + 1)] = np.vstack((ac[1:yeval[b_i].shape[0]]))
        # stack all the blocks
        rhs = np.vstack((rhsblk))
        lhs = np.vstack((lhsblk))
        # I can solve with the U method in SVD
        sol = self.svd_solution(lhs, rhs)
        self.B = np.reshape((sol[:self.n * self.m, :]),
                            (self.n, self.m), order='F')
        self.D = np.zeros((self.n_obs, self.m))

    def recomputeGamma(self):
        # Assign the new C to the Gamma matrix
        gam_arr = [self.C for _ in range(self.hl_bl)]
        # iteratively multiply by A
        for blk in range(1, self.hl_bl):
            gam_arr[blk] = gam_arr[blk - 1] @ self.A
        self.Gamma = np.vstack((gam_arr))

    def fullGamma(self, samples):
        gam_arr = [self.C for _ in range(samples)]
        for s in range(1, samples):
            gam_arr[s] = gam_arr[s - 1] @ self.A
        return gam_arr

    def oblique_prj(self, X, Y, Z):
        # Oi = X/_{Y}(Z) the oblique projection of X, onto
        # the row space of Z along Y. For that,
        # Oi = X/_{Y}(Z) = (X/oY)*pinv(Z/oY)*Z,
        # where the (X/oY) notation is the projection of Y onto the
        # orthogonal complement of Y

        # Yf/oUf Y_future projected onto the otrthogonal complement of U_future
        yfPOuf = self.z_proj_O_x(X, Y)
        # Wp/oUf Wp=[Up;Yp] The instrumental variable projected onto the
        # orthogonal complement of U_future
        wpPOuf = np.linalg.pinv(
            self.z_proj_O_x(Z, Y)
        )  # return the oblique projection
        return yfPOuf @ wpPOuf @ Z

    def xu_eval(self, system):
        # for sid, just evaluate the outputs
        obsrv = self.observable.obs_fun()
        ytr = [np.squeeze(obsrv(*sp["y"].T).T) for sp in system]
        if "u" in system[0].keys():
            utr = [sp["u"] for sp in system]
        else:
            utr = []

        return ytr, utr

    def predict(self, x0, n_points, system=[]):
        # obs = self.observable.obs_fun()
        # preallocate
        prediction = [
            {"y": np.zeros((n_p, self.observable.l)),
             "x": np.zeros((n_p, self.n))}
            for n_p in n_points
        ]
        for sample, x0_s in enumerate(x0):
            prediction[sample]["y"][0, :] = (self.edmdC @ self.C @ x0_s).T
            prediction[sample]['x'][0, :] = x0_s.T
            for step in range(1, n_points[sample]):
                x_prev = prediction[sample]['x'][step - 1, :].T
                input = system[sample]["u"][step - 1, :].T
                prediction[sample]['x'][step,
                                        :] = self.A @ x_prev + self.B @ input
                prediction[sample]["y"][step,
                                        :] = self.edmdC @ (self.C @ prediction[sample]['x'][step, :].T + self.D @ input)

        return prediction

    def predict_from_test(self, system):
        # Predicts form a testing dicionary with possibly, many samples
        x0 = [self.initial_condition(sys_i) for sys_i in system]
        # number of points per simulation
        n_p = [sys_i['y'].shape[0] for sys_i in system]
        pred_test = self.predict(x0, n_p, system)
        return pred_test

    def initial_condition(self, sys_i):
        obs = self.observable.obs_fun()
        # number of sample points to include in the optim
        points = self.hl_bl
        # get the gamma matrix in individual blocks
        ca = self.fullGamma(points)
        # multiply each block by B
        cab = [ca[i] @ self.B for i, ca_i in enumerate(ca)]
        # now I need to sum per sample. Each entry is the same size as cab
        cabu = list(cab)  # This is just preallocation
        for y_k in range(points):
            cabuk = np.zeros((self.n_obs, y_k + 1))
            for r in range(y_k + 1):
                cabuk[:, r] = cab[y_k - r] @ sys_i["u"][r, :]
            cabu[y_k] = np.sum(cabuk, axis=1, keepdims=True)

        # build the right hand side and the left hand side
        lhs = np.reshape(np.squeeze(obs(*sys_i['y'][1:points].T).T),
                         [-1, 1], order='F') - np.vstack((cabu[:points - 1]))
        rhs = np.vstack(ca[1:points])
        x0 = self.svd_solution(lhs, rhs)
        # little test
        # y_i0 = self.edmdC @ self.C @ x0
        # x0 = np.linalg.pinv(self.C) @ obs(*sys_i['y'][0, :].T)
        # y_i = sys_i['y'][0, :]
        # # self.spectrum()
        # # plt.show()
        return x0

    def hankel_blocks(self, yeval):
        # Samples per trajectory get the lenght of all the trajectories in the
        # training set
        # I only need the minimum to set the hankel blocks
        min_spl = np.min([sp.shape[0] for sp in yeval])
        # maximize the number of blocks while keeping the hankel matrix fat
        self.hl_bl = int(np.floor((min_spl + 1) / (2 * (self.n_obs + 1))))

    @ staticmethod
    def block_hankel(mat, s):
        # mat is the matrix to hankelize,
        # s is the number of blocks
        j_samples = mat.shape[1]
        # number of columns in the final matrix
        n_col = int(j_samples - 2 * s + 1)
        # Assign the values
        # This can be a one liner
        return np.vstack([mat[:, blk: blk + n_col] for blk in range(s)])

    @ staticmethod
    def z_proj_O_x(z, x):
        # z projected onto the row space of  the orthogonal complement of x
        p = z.shape[0]
        Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        R = R.T
        return R[-p:, -p:] @ Q[:, -p:].T

    @ staticmethod
    def z_proj_x(z, x):
        # z projected onto the row space of x
        p = z.shape[0]
        Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        R = R.T
        return R[-p:, :-p] @ Q[:, :-p].T
    # def linear_sol_ABCD(self, Ysid, Usid):
    #     # # Needs refactoring... too long and unreadable... # #
    #     # From the calculated Gamma, and the hankel matrices,
    #     # Get the A, B, C and D matrices
    #     # The Gamma minus mat
    #     gmm = self.Gamma[:-self.n_obs, :]
    #     # Inverses of the gammas
    #     gam_inv = np.linalg.pinv(self.Gamma)
    #     gmm_inv = np.linalg.pinv(gmm)
    #
    #     # Linear equations for A, C
    #     # z_i = Yf/[Wp;Uf]
    #     wp = np.vstack((
    #         Usid[:self.hl_bl * self.m, :],  # U_past
    #         Ysid[:self.hl_bl * self.n_obs, :]  # Y_past
    #     ))
    #     z_i = self.z_proj_x(
    #         Ysid[self.hl_bl * self.n_obs:, :],  # Y_future
    #         np.vstack((
    #             wp,
    #             Usid[self.hl_bl * self.m:, :]  # U_future
    #         ))
    #     )
    #     # zip = Yf-/[Wp+;Uf-]
    #     wpp = np.vstack((
    #         Usid[:(self.hl_bl + 1) * self.m, :],  # U_past+
    #         Ysid[:(self.hl_bl + 1) * self.n_obs, :]  # Y_past+
    #     ))
    #     zip = self.z_proj_x(
    #         Ysid[(self.hl_bl + 1) * self.n_obs:, :],
    #         np.vstack((
    #             wpp,
    #             Usid[(self.hl_bl + 1) * self.m:, :]  # U_future+
    #         ))
    #     )
    #     # left-hand-side = (gmm_inv*zip;Yii)
    #     lhs = np.vstack((
    #         gmm_inv @ zip,
    #         Ysid[self.hl_bl * self.n_obs:(self.hl_bl + 1) * self.n_obs, :]
    #     ))
    #     # right-hand-side = (gam_inv*z_i;Uf)
    #     rhs = np.vstack((
    #         gam_inv @ z_i,
    #         Usid[self.hl_bl * self.m:, :]  # U_future
    #     ))
    #     AC = lhs @ np.linalg.pinv(rhs)
    #     # I do not need to compute and then extract.
    #     self.A = AC[:self.n, :self.n]
    #     self.C = AC[self.n:, :self.n]
    #     # Calculate the residuals
    #     res = lhs - AC @ rhs
    #
    #     # From A and C, recompute Gamma
    #     self.recomputeGamma()
    #     # And get the Gamma- and the inverses again
    #     gmm = self.Gamma[:-self.n_obs, :]
    #     # Inverses of the gammas
    #     gam_inv = np.linalg.pinv(self.Gamma)
    #     gmm_inv = np.linalg.pinv(gmm)
    #
    #     # Here I am following the alg from the book. I do not understand it
    #     # completely yet. pg: 125
    #
    #     P = lhs - np.vstack((self.A, self.C)) @ rhs[:self.n, :]
    #     Q = Usid[self.hl_bl * self.m:, :]  # U_future
    #
    #     L1 = self.A @ gam_inv
    #     L2 = self.C @ gam_inv
    #
    #     M = np.hstack((np.zeros((self.n, self.n_obs)), gmm_inv))
    #     X = np.vstack((
    #         np.hstack((np.eye(self.n_obs), np.zeros((self.n_obs, self.n)))),
    #         np.hstack((np.zeros((self.n_obs * (self.hl_bl - 1), self.n_obs)), gmm))
    #     ))
    #
    #     totm = 0
    #     for k in range(self.hl_bl):
    #         N = np.vstack((
    #             np.hstack((M[:, k * self.n_obs:] - L1[:, k * self.n_obs:],
    #                        np.zeros((self.n, k * self.n_obs)))),
    #             np.hstack((-L2[:, k * self.n_obs:],
    #                        np.zeros((self.n_obs, k * self.n_obs))))
    #         ))
    #         if k == 0:
    #             N[self.n:, :self.n_obs] = np.eye(self.n_obs) + N[self.n:, :self.n_obs]
    #
    #         N = N @ X
    #         totm += np.kron(Q[k * self.m:(k + 1) * self.m, :].T, N)
    #
    #     bd_vec = np.linalg.lstsq(totm, P.reshape((-1, 1), order='F'))[0]
    #     bd = bd_vec.reshape((self.n + self.n_obs, self.m), order='F')
    #     # Different from matlab, I can mofify the object here
    #     self.B = bd[:self.n_obs, :]
    #     self.D = bd[self.n_obs:self.n_obs + self.n, :]
    #     return res
