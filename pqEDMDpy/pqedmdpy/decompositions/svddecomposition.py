"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on the singular value decomposition
"""

import sys

import numpy as np

from pqedmdpy.decompositions import pqdecomposition


class svdDecomposition(pqdecomposition.pqDecomposition):
    def __init__(
        self,
        observable,
        system,
    ):
        # Class to perform the regression 
        # and use the SVD as the inverse engine
        self.observable = observable
        o_pst, o_fut = self.y_snapshots(system)

        # After calculating the past and future,
        # We need to check for inputs
        if "u" in system[0].keys():
            # if there is an input to the system
            self.m = np.shape(system[0]["u"])[1]
            u_pst, u_fut = self.u_snapshots(system)
            o_pst = np.concatenate((o_pst, u_pst), axis=1)
            o_fut = np.concatenate((o_fut, u_fut), axis=1)
        else:
            self.m = 0

        self.l = self.observable.l
        self.n_obs = np.shape(self.observable.pq_mat())[1] + 1
        # Perform the SVD of the past data
        Ud, S, V = np.linalg.svd(o_pst)
        # get the effective rank
        r = self.effective_rank(S, o_pst)
        # trim the matrices
        Ur = Ud[:, :r]
        Sr = np.diag(S[:r])
        Vr = V[:, :r]

        Dr = np.linalg.lstsq(Sr, Ur.T @ o_fut, rcond=None)[0]

        U = Vr.T @ Dr

        self.A = self.matrix_A(U)
        self.B = self.matrix_B(U)
        self.C = self.matrix_C()
        self.D = np.zeros((self.l, 1))

    def effective_rank(self, s, vareval):
        '''
        s is the list of singular values
        vareval is the matrix to evaluate.
        '''
        r = 1
        while (r <= self.n_obs) and (
            s[r] > max(vareval.shape) * sys.float_info.epsilon * s[0]
        ):
            r += 1
        return r
