"""jkjkjkjkjkj
Author: Camilo Garcia Tenorio
returns a pq decomposition object
based on subspace identification
"""

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pqobservable import pqObservable
from decompositions.svddecomposition import svdDecomposition


class sidDecomposition(svdDecomposition):
    """ Decomposition method for EDMD based in subspace identification

    Attributes:
        pb          Past Hankel Blocks
        fb          Future Hankel Blocks
        unforced    Unforced or not
        det         Deterministic true or false
        K           Kalman
        Cedmd       C matrix to return from the function space
    """

    def __init__(
        self,
        fb: int,  # Future Hankel blocks
        pb: int,  # Past Hankel blocks
        observable: pqObservable,
        system,
    ):
        # Class to perform the regression
        # and use the SVD as the inverse engine
        self.observable = observable
        if "u" in system[0]:
            # if there is an input to the system
            self.sys_m = np.shape(system[0]["u"])[1]
            # We need at least one past block if the system is forced
            self.unforced = False
            if (pb != 0):
                self.det = False
                self.pb = pb
            else:
                self.det = True
                # At least one block if it its forced and deterministic
                self.pb = 1
        else:
            self.sys_m = 0
            self.unforced = True
            self.pb = pb
            if (pb != 0):
                self.det = False
            else:
                self.det = True

        self.num_obs: int = np.shape(self.observable.pq_mat())[1]
        self.fb = fb
        self.sys_l: int = self.observable.obs_l  # Copy of the obs attr

        # Evaluate the data
        xeval, u_sys = self.xu_eval(system)

        Ysid = [self.block_hankel(x.T, self.fb, self.pb) for x in xeval]
        if self.unforced:
            Usid = [np.zeros((2*self.fb, y_i.shape[1])) for y_i in Ysid]
        else:
            Usid = [self.block_hankel(u.T, self.fb, self.pb) for u in u_sys]

        self.A, self.B, self.C, self.D, self.K, self.sys_n = self.ABCDKn(
            Ysid, Usid)
        # From the new A and C, recompute Gamma
        # Calculate the C that brings back the espanded state to the output
        self.Cedmd: np.ndarray = self.matrix_C()[:, 1:]

    def ABCDKn(self, Ysid, Usid) -> None:
        U, S = self.compute_svd(Ysid, Usid)
        ns = np.arange(self.num_obs, sum(S > 1e-9)+1, 1)
        nv = [self.n_cost(n, U, S, Ysid, Usid)[0] for n in ns]
        n = ns[np.argmin(nv)]
        _, res, a, b, c, d = self.n_cost(n, U, S, Ysid, Usid)
        sigWE = res@res.T/(res.shape[1]-1)
        QQ = sigWE[:n, :n]
        RR = sigWE[n:n+self.num_obs, n:n+self.num_obs]
        SS = sigWE[:n, n:n+self.num_obs]
        try:
            x = dare(a.T, c.T, QQ, RR, None, SS)
            k = (np.linalg.inv(c@x@c.T+RR)@(c@x@a.T+SS.T)).T
        except Exception:
            k = np.zeros((n, self.sys_l))

        # in the unforced case, we need b and d to be zero and not empty
        if self.sys_m == 0:
            b = np.zeros((n, 1))
            d = np.zeros((self.num_obs, 1))
        return a, b, c, d, k, n

    def compute_svd(self, Ysid, Usid):
        y_f, u_f, w_p = self.fut_pst_mat(Ysid, Usid)
        if self.unforced:
            if self.det:
                WOW = np.hstack((y_f))
            else:
                WOW = np.hstack((
                    [self.z_proj_x(yi, wpi) for yi, wpi in zip(y_f, w_p)]
                ))
        else:
            Oi = [self.obliqueP(yi, ui, wpi)
                  for yi, ui, wpi in zip(y_f, u_f, w_p)]
            WOW = np.hstack((
                [self.z_projOx(oii, ui) for oii, ui in zip(Oi, u_f)]
            ))
        # At last, compute the svd
        U, S, _ = np.linalg.svd(WOW)
        return U, S

    def n_cost(self, n, U, S, Ysid, Usid):
        # Only the inverse of gamma is necessary
        gam_inv = np.linalg.pinv(U[:, : n] @ np.diag(np.sqrt(S[: n])))
        # Future and past matrices
        y_f, u_f, w_p = self.fut_pst_mat(Ysid, Usid)
        z_i = self.get_zi(y_f, u_f, w_p)
        # Calculate the state sequence
        x = [gam_inv@zii for zii in z_i]
        # For the solution, get the division like in the original formulation
        # of the EDMD
        # Get the divided matrices.
        xp, xf, yiip, ufp, uff = self.edmd_like_division(x, y_f, u_f)
        # Start with A and B
        ab_lhs = np.vstack((np.hstack((xf)), np.hstack((uff))))
        ab_rhs = np.vstack((np.hstack((xp)), np.hstack((ufp))))
        ab = self.regression(ab_lhs.T, ab_rhs.T).T
        # Extract the matrices
        a = ab[:n, :n]
        if self.sys_m == 0:
            b = np.zeros((n, 0))
        else:
            b = ab[: n, n:]
        # Continue with C and D
        c_lhs = np.hstack((yiip))
        c_rhs = np.hstack((xf))
        c = self.regression(c_lhs.T, c_rhs.T).T
        d = np.zeros((c.shape[0], b.shape[1]))
        # Residuals and cost
        res = np.vstack((np.hstack((xf)), np.hstack((yiip)))) - \
            np.vstack((np.hstack((a, b)), np.hstack((c, d))))@ab_rhs
        cost = np.sum(np.abs(res))
        return cost, res, a, b, c, d

    def get_zi(self, y_f, u_f, w_p):
        if self.unforced:
            if self.det:
                z_i = y_f
            else:
                z_i = [self.z_proj_x(yi, wi) for yi, wi in zip(y_f, w_p)]
        else:
            z_i = [self.obliqueP(yi, ui, wi)
                   for yi, ui, wi in zip(y_f, u_f, w_p)]
        return z_i

    def edmd_like_division(self, x, y_f, u_f):
        x_p = [xi[:, :-1] for xi in x]
        x_f = [xi[:, 1:] for xi in x]
        yii_p = [yi[:self.num_obs, :xi.shape[1]-1] for yi, xi in zip(y_f, x)]
        u_fp = [ui[:self.sys_m, :xi.shape[1]-1] for ui, xi in zip(u_f, x)]
        u_ff = [ui[:self.sys_m, 1:xi.shape[1]] for ui, xi in zip(u_f, x)]
        return x_p, x_f, yii_p, u_fp, u_ff

    def obliqueP(self, X, Y, Z):
        # Oi = X/_{Y}(Z) the oblique projection of X, onto
        # the row space of Z along Y. For that,
        # Oi = X/_{Y}(Z) = (X/oY)*pinv(Z/oY)*Z,
        # where the (X/oY) notation is the projection of Y onto the
        # orthogonal complement of Y

        # Yf/oUf Y_future projected onto the otrthogonal complement of U_future
        yfPOuf = self.z_projOx(X, Y)
        # Wp/oUf Wp=[Up;Yp] The instrumental variable projected onto the
        # orthogonal complement of U_future
        wpPOuf = np.linalg.pinv(
            self.z_projOx(Z, Y)
        )  # return the oblique projection
        return yfPOuf @ wpPOuf @ Z

    def xu_eval(self, system):
        # for sid, just evaluate the outputs
        obsf = self.observable.obs_fun()
        ytr = [np.squeeze(obsf(*sp["y"].T).T) for sp in system]
        if "u" in system[0].keys():
            utr = [sp["u"] for sp in system]
        else:
            utr = []

        return ytr, utr

    def predict(self, y0, n_points, u=[]):
        # Handle the unforced case
        if (self.sys_m == 0):  # There is no input
            u = [np.zeros((pts, 1)) for pts in n_points]

        obsf = self.observable.obs_fun()
        Cinv = np.linalg.pinv(self.C)
        # preallocate
        pred = [
            {"y": np.zeros((n_p, self.sys_l)),
             "sv": np.zeros((n_p, self.sys_n))}
            for n_p in n_points
        ]
        # Simulate
        for orb, y0_s in enumerate(y0):  # for all initial conditions
            pred[orb]["y"][0, :] = y0_s
            pred[orb]['sv'][0, :] = (
                Cinv@(obsf(*pred[orb]["y"][0, :]))).T
            for step in range(1, n_points[orb]):
                # Lift the previous output
                lft = obsf(*pred[orb]['y'][step-1, :])
                # Lifted to space state
                x_prev = Cinv@lft
                # Evolve
                x_post = self.A@x_prev + self.B@u[orb][step:step+1, :].T
                # Save the state
                pred[orb]['sv'][step, :] = x_post.T
                # Save the output
                pred[orb]['y'][step, :] = (
                    self.Cedmd@(self.C@x_post +
                                self.D @ u[orb][step:step+1, :].T)).T
        return pred

    def fut_pst_mat(self, Ysid, Usid) -> np.ndarray:
        '''
        Returns all the necessary matrices for the sid algorithms past and future
        '''
        # future outputs
        y_f = [y_i[-self.fb*self.num_obs:, :] for y_i in Ysid]
        # y_f = Ysid[self.hb * self.n_obs:, :]
        # u_f = Usid[self.hb * self.sys_m:, :]
        u_f = [u_i[-self.fb*self.sys_m:, :] for u_i in Usid]
        w_p = [np.vstack(
            (u_i[:self.pb*self.sys_m, :], y_i[:self.pb*self.num_obs, :])
        ) for u_i, y_i in zip(Usid, Ysid)]
        return y_f, u_f, w_p

    @staticmethod
    def block_hankel(mat, fb, pb):
        # mat is the matrix to hankelize,
        # fb is the number of future blocks
        # pb is the number of past blocks
        # number of columns in the final matrix
        n_col = int(mat.shape[1] - fb - pb + 1)
        # Assign the values
        # This cnp.vstack([mat[:, blk: blk + n_col] for blk in range(2 * s)])an be a one liner
        return np.vstack([mat[:, blk: blk + n_col] for blk in range(fb+pb)])

    @staticmethod
    def z_projOx(z, x) -> np.matrix:
        # z projected onto the row space of  the orthogonal complement of x
        return z - z @ x.T @ np.linalg.pinv(x @ x.T) @ x

    @staticmethod
    def z_proj_x(z, x):
        # z projected onto the row space of x
        return z @ x.T @ np.linalg.pinv(x @ x.T) @ x

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
    # def BD(self, yeval, u):
    #     # Check first if the system is forced
    #     if ~self.m:
    #         self.B = np.zeros((self.n, 1))
    #         self.D = np.zeros((self.n_obs, 1))
    #         return
    #
    #     # I need the maximun samples in all the trajectories
    #     max_sam = np.max([np.shape(y_i)[0] for y_i in yeval])
    #     ac = [self.C for _ in range(max_sam)]
    #     for ac_blk in range(1, max_sam):
    #         ac[ac_blk] = ac[ac_blk - 1] @ self.A
    #
    #     ukrac = [
    #         [np.kron(np.zeros((1, u_i.shape[1])), self.C)
    #          for _ in range(u_i.shape[0])]
    #         for u_i in u
    #     ]
    #     for u_in, u_i in enumerate(u):
    #         for u_blk in range(u_i.shape[0]):
    #             krcb = np.zeros((*ukrac[u_in][u_blk].shape, u_blk + 1))
    #             for step in range(u_blk + 1):
    #                 krcb[:, :, step] = np.kron(
    #                     u[u_in][step, :], ac[u_blk - step])
    #             ukrac[u_in][u_blk] = np.sum(krcb, axis=2)
    #     # Ok, kroneckers done... jumadre
    #     # Now I need the right-hand-side and the lhs
    #     rhsblk = [
    #         np.zeros((
    #             y_i[1:, :].size, self.n * (len(yeval) + self.m)
    #         )) for y_i in yeval
    #     ]
    #     lhsblk = [y_i[1:, :].T.reshape((-1, 1), order='F') for y_i in yeval]
    #     # Populate the rhsblk
    #     for b_i, _ in enumerate(rhsblk):
    #         rhsblk[b_i][:, : self.n * self.m] = np.vstack((ukrac[b_i][:-1]))
    #         rhsblk[b_i][:, self.n * (b_i + self.m): self.n *
    #                     (b_i + self.m + 1)] = np.vstack((ac[1:yeval[b_i].shape[0]]))
    #     # stack all the blocks
    #     rhs = np.vstack((rhsblk))
    #     lhs = np.vstack((lhsblk))
    #     # I can solve with the U method in SVD
    #     sol = self.svd_solution(lhs, rhs)
    #     self.B = np.reshape((sol[:self.n * self.m, :]),
    #                         (self.n, self.m), order='F')
    #     self.D = np.zeros((self.n_obs, self.m))

    # def hankel_blocks(self, yeval):
    #     # Samples per trajectory get the lenght of all the trajectories in the
    #     # training set
    #     # I only need the minimum to set the hankel blocks
    #     min_spl = np.min([sp.shape[0] for sp in yeval])
    #     # maximize the number of blocks while keeping the hankel matrix fat
    #     # The max line and autoformatters in python are annoying
    #     num_hlbl = (min_spl + 1) * len(yeval)
    #     den_hlbl = (2 * (self.n_obs + len(yeval)))
    #     self.hb = int(np.floor(num_hlbl / den_hlbl))
    # def compute_svd(self, Ysid, Usid):
    #     # Perform the projection
    #     y_f, u_f, w_p = self.fut_pst_mat(Ysid, Usid)
    #     Yf = Ysid[self.hb * self.n_obs:, :]
    #     Uf = Usid[self.hb * self.sys_m:, :]
    #     Wp = np.vstack(
    #         (
    #             Usid[: self.hb * self.sys_m, :],  # U_past
    #             Ysid[: self.hb * self.n_obs, :],  # Y_past
    #         )
    #     )
    #     Oi = self.obliqueP(Yf, Uf, Wp)
    #     WOW = self.z_projOx(Oi, Uf)
    #     # Finally, the singular value decomposition
    #     U, S, _ = np.linalg.svd(WOW)
    #     return U, S
    # def ACKn(self, Ysid, Usid):
    #     U, S = self.compute_svd(Ysid, Usid)
    #     # possible n's from the order of the system,
    #     # untill the criterion from some paper
    #     # ns = np.arange(
    #     #     self.l,
    #     #     2 * np.sum(np.log(S) > 0.5 * (np.log(S[0] + np.log(S[-1])))), 1
    #     # )
    #     max_n = np.min([self.hb, sum(S > 1e-5)])  # arbitrarily small...
    #     ns = np.arange(self.n_obs, max_n, 1)
    #     nv = [self.n_cost(ni, U, S, Ysid, Usid)[0] for ni in ns]
    #     # order of the system
    #     self.sys_n = ns[np.argmin(nv)]
    #     # Calculate everything
    #     _, res, self.A, self.C = self.n_cost(self.sys_n, U, S, Ysid, Usid)
    #     # over all possible values of n, calculate the error
    #     # Optimize the solution for the order of the System
    #     # I am missing K
    # def BD(self, Ysid, Usid):
    #     # Check first if the system is forced
    #     if (self.sys_m == 0):
    #         self.B = np.zeros((self.sys_n, 1))
    #         self.D = np.zeros((self.n_obs, 1))
    #         return
    #     # The las BD was not robust and the noise killed it
    #     # This one is robust because it does not go back to the data.
    #
    #     # There are two possible cases, either U_{future} is close to
    #     # singular, or not. Still, the two cases must return a Kc matrix
    #     # Everything depends on the gamma matrices, and their inverses.
    #     gam = self.Gamma
    #     gmm = gam[: -self.n_obs, :]
    #     gam_inv = np.linalg.pinv(gam)
    #     gmm_inv = np.linalg.pinv(gmm)
    #     # Also, we need the Yii, Uf, zip, and z_i matrices
    #     y_f, u_f, w_p, yfm, ufm, wpp = self.fut_pst_mat(Ysid, Usid)
    #     yii = y_f[:self.n_obs, :]
    #     # should I carry or recalculate? What a conundrum!!!
    #     z_i = self.z_proj_x(y_f, np.vstack((w_p, u_f)))
    #     zpi = self.z_proj_x(yfm, np.vstack((wpp, ufm)))
    #
    #     # Nk
    #     Nk = self.Nk(gmm, gam_inv, gmm_inv)
    #     # Now, Uk' the individual blocks of u_f
    #     Uk = [u_f[bl * self.sys_m:(bl + 1) * self.sys_m, :]
    #           for bl in range(self.hb)]
    #     # get the Kronecker sum, better known as the Right Hand Side
    #     sum_kron = sum([np.kron(ui.T, ni) for ui, ni in zip(Uk, Nk)])
    #     # Wee need Pee, vectorized; vec(P) = vec([gmm_inv*zip;yii]-[A;C]*gam_inv*z_i)
    #     vec_P = (np.vstack((gmm_inv @ zpi, yii)) - np.vstack((self.A,
    #                                                          self.C)) @ gam_inv @ z_i).reshape((-1, 1), order='F')
    #     # With the two sides, get the matrices
    #     # The svd_solution cannot handle the size of the matrix. Just use
    #     # the pseudo inverse.
    #     DB, _, _, _ = np.linalg.lstsq(vec_P, sum_kron, rcond=None)
    #     # BD = self.svd_solution(Kc, Nck)
    #     # Extract B and D
    #     self.B = np.reshape(DB[:, self.n_obs * self.sys_m:].T,
    #                         (self.sys_n, self.sys_m),
    #                         order='F')
    #     self.D = np.reshape(DB[:, :self.n_obs * self.sys_m].T,
    #                         (self.n_obs, self.sys_m),
    #                         order='F')
    #
    # def Nk(self, gmm, gam_inv, gmm_inv) -> list[np.ndarray]:
    #     """
    #     Returns an array of Nk matrices k=1,...,hl_bl
    #     """
    #     # I need the two L matrices, L=[L_{A};L_{C}] because they depend
    #     # on the A and C matrices respectively.
    #     # L=[A;C]gamma_inv
    #     La = self.A @ gam_inv
    #     Lc = self.C @ gam_inv
    #
    #     # Matrix Gi=[[I_{n_obs} 0];[0 gmm]]
    #     # Upper Gi
    #     Gup = np.hstack(
    #         (np.eye(self.n_obs), np.zeros((self.n_obs, self.sys_n))))
    #     # Lower stack of Gi
    #     Gbo = np.hstack(
    #         (np.zeros(((self.hb - 1) * self.n_obs, self.n_obs)), gmm))
    #     # Stack them Gi=[Gup;Gbo]
    #     Gi = np.vstack((Gup, Gbo))
    #
    #     # The Na matrices
    #     # Na1 = [-La,1 M1-La,2 M2-La,3 ... M{hb-1}-La,hb)] hb=hl_bl
    #     Na1 = np.hstack((
    #         -La[:, :self.n_obs], gmm_inv - La[:, self.n_obs:]
    #     )) @ Gi
    #     # All the remaining
    #     Nai = [np.hstack((
    #         gmm_inv[:, (bl - 1) * self.n_obs:] - La[:, bl * self.n_obs:],
    #         np.zeros((self.sys_n, bl * self.n_obs))
    #     )) @ Gi for bl in range(1, self.hb)]
    #     # Concatenate in Na
    #     Na = [Na1, *Nai]
    #     # The same for Nc
    #     # Start with the beginning, i.e., the first one, the first matrix. :P
    #     Nc1 = np.hstack((
    #         np.eye(self.n_obs) - Lc[:, :self.n_obs], Lc[:, self.n_obs:]
    #     )) @ Gi
    #     # Compute the remaining,
    #     Nci = [np.hstack((
    #         -Lc[:, bl * self.n_obs:],
    #         np.zeros((self.n_obs, bl * self.n_obs))
    #     )) @ Gi for bl in range(1, self.hb)]
    #     # concatenate
    #     Nc = [Nc1, *Nci]
    #     # Reconcatenate in Ni = [Nai;Nci]
    #     Nk = [np.vstack((nai, nci)) for nai, nci in zip(Na, Nc)]
    #     return Nk
    #
    # def Kc(self, yii, u_f, gam_inv, z_i) -> np.ndarray:
    #     """
    #     Returns the K_{c} stack of matrices from the robust ID alg.
    #     There are two cases, when u_f*u_f' is well behaved, i.e.,
    #     has a good condition number, and when it is not.
    #     """
    #     rc_threshold = 1e-40
    #     if (1 / np.linalg.cond(u_f @ u_f.T) < rc_threshold):
    #         # ill-conditioned
    #         lhs = (yii - self.C @ gam_inv @ z_i).T
    #         rhs = u_f.T
    #         Kc = self.svd_solution(lhs, rhs).T
    #     else:
    #         Kc = (yii - self.C @ gam_inv @ z_i) @ np.linalg.pinv(u_f)
    #
    #     Kc_arr = np.vstack(([Kc[:, bl * self.sys_m:(bl + 1) * self.sys_m]
    #                        for bl in range(self.hb)]))
    #     # Not working... The inverse of u_f fucks up everything
    #     return Kc_arr
    #
    # def initial_condition(self, sys_i):
    #     #   if ~isfield(sys,'u')
    #     #   sys.u = 0;
    #     # end
    #     # % After a lot of headaches, the best is to inverse the measurement y
    #     # x0 = pinv(obj.C)*(pinv(obj.C_edmd)*sys.y(1,:)'-obj.D*sys.u(1,:)');
    #     if ~("u" in sys_i):
    #         sys_i['u'] = np.array([[0]])
    #
    #     x0 = np.linalg.pinv(self.C) @ (np.linalg.pinv(self.Cedmd)
    #                                    @ sys_i["y"][0:1, :].T - self.D @ sys_i['u'][0:1, :].T)
    #     # This is the result of keeping it simple...
    #     return x0
    # def predict_from_test(self, system):
    #     # Predicts form a testing dicionary with possibly, many samples
    #     # x0 = [self.initial_condition(sys_i) for sys_i in system]
    #     # number of points per simulation
    #     n_p = [sys_i['y'].shape[0] for sys_i in system]
    #     pred_test = self.predict(x0, n_p, system)
    #     return pred_test
    # def recomputeGamma(self):
    #     # Assign the new C to the Gamma matrix
    #     gam_arr = [self.C for _ in range(self.fb)]
    #     # iteratively multiply by A
    #     for blk in range(1, self.fb):
    #         gam_arr[blk] = gam_arr[blk - 1] @ self.A
    #     self.Gamma = np.vstack((gam_arr))
