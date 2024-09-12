"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
based on the singular value decomposition
"""

import sys
import numpy as np

from decompositions.pqdecomposition import pqDecomposition


class svdDecomposition(pqDecomposition):
    """
    Class for performing a decomposition to a particular system based on a set
    of observables. For this one, the regression motor is an svd decomposition with effective rank

    Attributes
    A : Evolution matrix ψ(x)_+ = A ψ(x) + B u
    B : Input matrix       this one  ^
    C : Output matrix y = C ψ + D u
    D : Input to output matrix  ^ this one
    sys_l : number of outputs
    sys_m : number of inputs
    sys_n : order of the system
    num_obs : number of observables
    """

    def regression(self, lhs, rhs):
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
