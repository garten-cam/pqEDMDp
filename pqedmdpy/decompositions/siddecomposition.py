"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on subspace identification
"""

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pqedmdpy.decompositions import svddecomposition
from pqedmdpy.pqObservable import pqObservable


class sidDecomposition(svddecomposition.svdDecomposition):
    def __init__(
        self,
        observable: pqObservable,
        system,
    ):
        # Class to perform the regression
        # and use the SVD as the inverse engine
        self.observable = observable
        # Evaluate the data
        xeval, u = self.xu_eval(system)
        if "u" in system[0].keys():
            # if there is an input to the system
            n_in = np.shape(system[0]["u"])[1]
        else:
            n_in = 0

        self.sys_m: int = n_in
        self.n_obs: int = np.shape(self.observable.pq_mat())[1]
        self.sys_l: int = self.observable.obs_l
        self.hankel_blocks(xeval)

        Ysid = np.hstack(
            [self.block_hankel(per_x.T, 2 * self.hl_bl) for per_x in xeval]
        )
        if "u" in system[0].keys():
            # if there is an input to the system
            Usid = np.hstack([self.block_hankel(per_u.T, 2 * self.hl_bl)
                              for per_u in u])
        else:
            Usid = np.zeros((2 * self.hl_bl, Ysid.shape[1]))
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
        self.BD(Ysid, Usid)
        self.edmdC: np.ndarray = self.matrix_C()[:, 1:]

    def compute_svd(self, Ysid, Usid):
        # Perform the oblique projection
        # The last approach is not extendable. Refactoring oblique P
        Yf = Ysid[self.hl_bl * self.n_obs:, :]
        Uf = Usid[self.hl_bl * self.sys_m:, :]
        Wp = np.vstack(
            (
                Usid[: self.hl_bl * self.sys_m, :],  # U_past
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
        # ns = np.arange(
        #     self.l,
        #     2 * np.sum(np.log(S) > 0.5 * (np.log(S[0] + np.log(S[-1])))), 1
        # )
        max_n = np.min([self.hl_bl, sum(S > 1e-5)])  # arbitrarily small...
        ns = np.arange(self.n_obs, max_n, 1)
        nv = [self.n_cost(ni, U, S, Ysid, Usid)[0] for ni in ns]
        # order of the system
        self.sys_n = ns[np.argmin(nv)]
        # Calculate everything
        _, res, self.A, self.C = self.n_cost(self.sys_n, U, S, Ysid, Usid)
        # over all possible values of n, calculate the error
        # Optimize the solution for the order of the System
        # I am missing K

    def n_cost(self, n, U, S, Ysid, Usid):
        # Truncate according to the order
        Un = U[:, : n]
        Sn = S[: n]
        # Gamma, gamma_minus and their inverses
        gam = Un @ np.diag(np.sqrt(Sn))
        gmm = gam[: -self.n_obs, :]
        gam_inv = np.linalg.pinv(gam)
        gmm_inv = np.linalg.pinv(gmm)

        # slice the Hankel matrices
        y_f, u_f, w_p, yfm, ufm, wpp = self.fut_pst_mat(Ysid, Usid)

        # Get the z matrices
        # z_i = Yf/[wp;Uf]
        z_i = self.z_proj_x(y_f, np.vstack((w_p, u_f)))
        # zip = Yf-/[Wp+;Uf-]
        zip = self.z_proj_x(yfm, np.vstack((wpp, ufm)))

        # prepare for regression
        # left-hand-side = (gmm_inv*zip;Yii); Yii: first block of y_f
        lhs = np.vstack((gmm_inv @ zip, y_f[:self.n_obs, :]))
        # right-hand-side = (gam_inv*z_i;Uf)
        rhs = np.vstack((gam_inv @ z_i, u_f))
        # lhs = [ac';k(bd)]*rhs
        # [ac';k(bd)] = lhs @ np.linalg.pinv(rhs)
        # instead of pinv, use the effective svd solution
        ac = self.svd_solution(lhs.T, rhs.T).T
        # Extract the matrices
        a = ac[: n, : n]
        c = ac[n:, : n]
        # Calculate the residuals
        res = lhs - ac @ rhs
        cost = np.sum(np.abs(res))
        return cost, res, a, c

    def BD(self, Ysid, Usid):
        # Check first if the system is forced
        if (self.sys_m == 0):
            self.B = np.zeros((self.sys_n, 1))
            self.D = np.zeros((self.n_obs, 1))
            return
        # The las BD was not robust and the noise killed it
        # This one is robust because it does not go back to the data.

        # There are two possible cases, either U_{future} is close to
        # singular, or not. Still, the two cases must return a Kc matrix
        # Everything depends on the gamma matrices, and their inverses.
        gam = self.Gamma
        gmm = gam[: -self.n_obs, :]
        gam_inv = np.linalg.pinv(gam)
        gmm_inv = np.linalg.pinv(gmm)
        # Also, we need the Yii, Uf, zip, and z_i matrices
        y_f, u_f, w_p, yfm, ufm, wpp = self.fut_pst_mat(Ysid, Usid)
        yii = y_f[:self.n_obs, :]
        # should I carry or recalculate? What a conundrum!!!
        z_i = self.z_proj_x(y_f, np.vstack((w_p, u_f)))
        zpi = self.z_proj_x(yfm, np.vstack((wpp, ufm)))

        # Nk
        Nk = self.Nk(gmm, gam_inv, gmm_inv)
        # Now, Uk' the individual blocks of u_f
        Uk = [u_f[bl * self.sys_m:(bl + 1) * self.sys_m, :]
              for bl in range(self.hl_bl)]
        # get the Kronecker sum, better known as the Right Hand Side
        sum_kron = sum([np.kron(ui.T, ni) for ui, ni in zip(Uk, Nk)])
        # Wee need Pee, vectorized; vec(P) = vec([gmm_inv*zip;yii]-[A;C]*gam_inv*z_i)
        vec_P = (np.vstack((gmm_inv @ zpi, yii)) - np.vstack((self.A,
                 self.C)) @ gam_inv @ z_i).reshape((-1, 1), order='F')
        # With the two sides, get the matrices
        # The svd_solution cannot handle the size of the matrix. Just use
        # the pseudo inverse.
        DB, _, _, _ = np.linalg.lstsq(vec_P, sum_kron, rcond=None)
        # BD = self.svd_solution(Kc, Nck)
        # Extract B and D
        self.B = np.reshape(DB[:, self.n_obs * self.sys_m:].T,
                            (self.sys_n, self.sys_m),
                            order='F')
        self.D = np.reshape(DB[:, :self.n_obs * self.sys_m].T,
                            (self.n_obs, self.sys_m),
                            order='F')

    def Nk(self, gmm, gam_inv, gmm_inv) -> list[np.ndarray]:
        """
        Returns an array of Nk matrices k=1,...,hl_bl
        """
        # I need the two L matrices, L=[L_{A};L_{C}] because they depend
        # on the A and C matrices respectively.
        # L=[A;C]gamma_inv
        La = self.A @ gam_inv
        Lc = self.C @ gam_inv

        # Matrix Gi=[[I_{n_obs} 0];[0 gmm]]
        # Upper Gi
        Gup = np.hstack(
            (np.eye(self.n_obs), np.zeros((self.n_obs, self.sys_n))))
        # Lower stack of Gi
        Gbo = np.hstack(
            (np.zeros(((self.hl_bl - 1) * self.n_obs, self.n_obs)), gmm))
        # Stack them Gi=[Gup;Gbo]
        Gi = np.vstack((Gup, Gbo))

        # The Na matrices
        # Na1 = [-La,1 M1-La,2 M2-La,3 ... M{hb-1}-La,hb)] hb=hl_bl
        Na1 = np.hstack((
            -La[:, :self.n_obs], gmm_inv - La[:, self.n_obs:]
        )) @ Gi
        # All the remaining
        Nai = [np.hstack((
            gmm_inv[:, (bl - 1) * self.n_obs:] - La[:, bl * self.n_obs:],
            np.zeros((self.sys_n, bl * self.n_obs))
        )) @ Gi for bl in range(1, self.hl_bl)]
        # Concatenate in Na
        Na = [Na1, *Nai]
        # The same for Nc
        # Start with the beginning, i.e., the first one, the first matrix. :P
        Nc1 = np.hstack((
            np.eye(self.n_obs) - Lc[:, :self.n_obs], Lc[:, self.n_obs:]
        )) @ Gi
        # Compute the remaining,
        Nci = [np.hstack((
            -Lc[:, bl * self.n_obs:],
            np.zeros((self.n_obs, bl * self.n_obs))
        )) @ Gi for bl in range(1, self.hl_bl)]
        # concatenate
        Nc = [Nc1, *Nci]
        # Reconcatenate in Ni = [Nai;Nci]
        Nk = [np.vstack((nai, nci)) for nai, nci in zip(Na, Nc)]
        return Nk

    def Kc(self, yii, u_f, gam_inv, z_i) -> np.ndarray:
        """
        Returns the K_{c} stack of matrices from the robust ID alg.
        There are two cases, when u_f*u_f' is well behaved, i.e.,
        has a good condition number, and when it is not.
        """
        rc_threshold = 1e-40
        if (1 / np.linalg.cond(u_f @ u_f.T) < rc_threshold):
            # ill-conditioned
            lhs = (yii - self.C @ gam_inv @ z_i).T
            rhs = u_f.T
            Kc = self.svd_solution(lhs, rhs).T
        else:
            Kc = (yii - self.C @ gam_inv @ z_i) @ np.linalg.pinv(u_f)

        Kc_arr = np.vstack(([Kc[:, bl * self.sys_m:(bl + 1) * self.sys_m]
                           for bl in range(self.hl_bl)]))
        # Not working... The inverse of u_f fucks up everything
        return Kc_arr

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
        # Handle the unforced case
        if (self.sys_m == 0):  # There is no input
            u = [np.zeros((pts, 1)) for pts in n_points]
        else:
            # Extract the input from the system variable
            u = [sample["u"] for sample in system]

        # preallocate
        prediction = [
            {"y": np.zeros((n_p, self.observable.obs_l)),
             "x": np.zeros((n_p, self.sys_n))}
            for n_p in n_points
        ]
        for sample, x0_s in enumerate(x0):
            prediction[sample]["y"][0, :] = (self.edmdC @ self.C @ x0_s).T
            prediction[sample]['x'][0, :] = x0_s.T
            for step in range(1, n_points[sample]):
                x_prev = prediction[sample]['x'][step - 1, :].T
                input = u[sample][step - 1, :].T
                prediction[sample]['x'][step,
                                        :] = self.A @ x_prev + self.B @ input
                prediction[sample]["y"][step,
                                        :] = self.edmdC @ ((self.C @ prediction[sample]['x'][step, :].T).T + (self.D @ input).T)

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
        if ~self.sys_m:
            cabu = [np.zeros((self.n_obs, 1)) for _ in ca]
        else:
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
        __import__('ipdb').set_trace()
        # little test
        # y_i0 = self.edmdC @ self.C @ x0
        # x0 = np.linalg.pinv(self.C) @ obs(*sys_i['y'][0, :].T)
        # y_i = sys_i['y'][0, :]
        # # self.spectrum()
        # # plt.show()
        # All the algorithms depend on this working correctly
        return x0

    def hankel_blocks(self, yeval):
        # Samples per trajectory get the lenght of all the trajectories in the
        # training set
        # I only need the minimum to set the hankel blocks
        min_spl = np.min([sp.shape[0] for sp in yeval])
        # maximize the number of blocks while keeping the hankel matrix fat
        # The max line and autoformatters in python are annoying
        num_hlbl = (min_spl + 1) * len(yeval)
        den_hlbl = (2 * (self.n_obs + len(yeval)))
        self.hl_bl = int(np.floor(num_hlbl / den_hlbl))

    def fut_pst_mat(self, Ysid, Usid) -> np.ndarray:
        '''
        Returns all the necessary matrices for the sid algorithms
        x_past, x_future, the minus versions that are shifted one hankel
        block and the instrumental variables w_p and wpp
        variant
        '''

        y_f = Ysid[self.hl_bl * self.n_obs:, :]  # Y_{future}: Y_{i:2i-1}
        u_f = Usid[self.hl_bl * self.sys_m:, :]  # U_{future}: U_{i:2i-1}
        y_p = Ysid[:self.hl_bl * self.n_obs, :]  # Y_{past}: Y_{0:i-1}
        u_p = Usid[:self.hl_bl * self.sys_m, :]  # U_{past}: U_{0:i-1}
        w_p = np.vstack((u_p, y_p))

        # Shifting one hankel block, the - matrices
        yfm = Ysid[(self.hl_bl + 1) * self.n_obs:, :]  # Y_{fut-}: Y_{i+1:2i-1}
        ufm = Usid[(self.hl_bl + 1) * self.sys_m:, :]  # U_{fut-}: U_{i+1:2i-1}
        ypp = Ysid[:(self.hl_bl + 1) * self.n_obs, :]  # Y_{pst+}: Y_{0:i}
        upp = Usid[:(self.hl_bl + 1) * self.sys_m, :]  # U_{pst+}: U_{0:i}
        wpp = np.vstack((upp, ypp))
        return y_f, u_f, w_p, yfm, ufm, wpp

    @ staticmethod
    def block_hankel(mat, s):
        # mat is the matrix to hankelize,
        # s is the number of blocks
        j_samples = mat.shape[1]
        # number of columns in the final matrix
        n_col = int(j_samples - s + 1)
        # Assign the values
        # This can be a one liner
        return np.vstack([mat[:, blk: blk + n_col] for blk in range(s)])

    @ staticmethod
    def z_proj_O_x(z, x) -> np.matrix:
        # z projected onto the row space of  the orthogonal complement of x
        # The projection thing
        # p = z.shape[0]
        # Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        # R = R.T
        # retv = R[-p:, -p:] @ Q[:, -p:].T <- this crap does not work
        return z - z @ x.T @ np.linalg.pinv(x @ x.T) @ x

    @ staticmethod
    def z_proj_x(z, x):
        # z projected onto the row space of x
        # p = z.shape[0]
        # Q, R = np.linalg.qr((np.vstack((x, z)).T), mode="reduced")
        # R = R.T
        # old = R[-p:, :-p] @ Q[:, :-p].T
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
