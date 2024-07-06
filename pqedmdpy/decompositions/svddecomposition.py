"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on the singular value decomposition
"""

import sys

import numpy as np

from pqedmdpy.decompositions import pqdecomposition
from pqedmdpy.pqObservable import pqObservable


class svdDecomposition(pqdecomposition.pqDecomposition):
    def __init__(
        self,
        observable: pqObservable,
        system: dict,
    ):
        # Class to perform the regression
        # and use the SVD as the inverse engine
        self.observable = observable
        o_pst, o_fut = self.y_snapshots(system)

        # After calculating the past and future,
        # We need to check for inputs
        if "u" in system[0].keys():
            # if there is an input to the system
            n_in = np.shape(system[0]["u"])[1]
            u_pst, u_fut = self.u_snapshots(system)
            o_pst = np.concatenate((o_pst, u_pst), axis=1)
            o_fut = np.concatenate((o_fut, u_fut), axis=1)
        else:
            n_in = 0

        self.sys_m: int = n_in
        self.sys_l = self.observable.obs_l
        self.n_obs = np.shape(self.observable.pq_mat())[1] + 1
        # Perform the SVD of the past data
        U = self.svd_solution(o_fut, o_pst)  # svd(lhs, rhs)

        self.A: np.ndarray = self.matrix_A(U)
        self.B: np.ndarray = self.matrix_B(U)
        self.C: np.ndarray = self.matrix_C()
        self.D: np.ndarray = np.zeros((self.sys_l, self.sys_m))

    def svd_solution(self, lhs, rhs):
        '''
        Solves the OLS lhs = sol * rhs
        '''
        Ud, S, V = np.linalg.svd(rhs)
        # get the effective rank
        r = svdDecomposition.effective_rank(S, rhs)
        # trim the matrices
        Ur = Ud[:, :r]
        Sri = np.diag(1 / (S[:r]))
        Vr = V[:r, :]
        return Vr.T @ Sri @ Ur.T @ lhs

    @staticmethod
    def effective_rank(s, vareval):
        '''
        s is the list of singular values
        vareval is the matrix to evaluate.
        '''
        r = np.sum(
            s > np.max((vareval.shape)) * sys.float_info.epsilon * s[0]
        )
        return r
