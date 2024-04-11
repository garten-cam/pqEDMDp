"""
Wrapper for the pqEDMD algorithm.
Author: Camilo Garcia Tenorio
"""

from pqedmdpy.pqObservable import legendreObs
from pqedmdpy.decompositions.pqdecomposition import pqDecomposition


class pqEDMDp:
    """
    A class to apply the decompositions to a system based on an array of
    observables.
    """

    def __init__(
        self,
        p=[1, 2],  # it can be an array of p parameters
        q=[0.9, 1.4],  # it can be an array of q parameters
        obs=legendreObs,
        dyn_dcp=pqDecomposition
    ):
        self.p = p
        self.q = q
        self.obs = obs
        self.dyn_dcp = dyn_dcp

    def fit(self, system):
        """
        Return an array of decompositions for all the p parameters, and all the
        p parameters, get a set of unique decompositions and fit them
        """
        vvfos = self.obs_list(system)
        # for each of the observables, calculate a decomposition
        return [self.dyn_dcp(obs, system) for obs in vvfos]

    def obs_list(self, system):
        """
        Returns a list of unique observables based on the provided p-q
        combinations.
        """
        # The system only provides the number of variables
        l = system[0]["y"].shape[1]  # assumes that the samples come in columns
        # list all unique vvfos
        vvfos = list(
            {self.obs(l=l, p=ip, q=iq) for ip in self.p for iq in self.q}
        )
        # This python way of defining a dictionary and assigning only unique
        # elements based on the eq function is nicer than matlab
        return vvfos
