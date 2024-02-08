"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
"""

import numpy as np
from matplotlib.pyplot import axis


class Decomposition:

    def __init__(
        self,
        observable,
        system,
    ):
        # General Decomposition class.
        # Implements the ordinary squares solution
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
        # Calculate
        g = np.matmul(o_pst.T, o_pst)
        a = np.matmul(o_pst.T, o_fut)
        U = np.matmul(np.linalg.inv(g), a)
        np.linalg.inv(g)

    def y_snapshots(self, sys):
        # function to evaluate the polynomials
        obsrv = self.observable.obs_fun()
        # Evaluate the Ys. In a system there are many samples. This function
        # takes the outputs "y" of the system and:
        # applies the function while slicing the past and future samples
        y_ob_pst = np.squeeze(
            np.concatenate([obsrv(*sp["y"][:-2, :].T).T for sp in sys], axis=0)
        )
        y_ob_fut = np.squeeze(
            np.concatenate([obsrv(*sp["y"][1:-1, :].T).T for sp in sys], axis=0)
        )
        return y_ob_pst, y_ob_fut

    def u_snapshots(self, sys):
        # This the same as the variables y without observation
        u_pst = np.concatenate([sp["u"][:-2, :] for sp in sys], axis=0)
        u_fut = np.concatenate([sp["u"][1:-1, :] for sp in sys], axis=0)
        return u_pst, u_fut
