# pqEDMD Implementation
'''
Author: Camilo Garcia Tenorio
Development of pqEDMD in python... lets see how this goes...
'''
from pqedmdpy import pqObservable as obs
from pqedmdpy import decompositions as dc
from itertools import product
import numpy as np


class pqEDMD:
    def __init__(self,
                 p=[2],  # it can be an array of p parameters
                 q=[0.9],  # it can be an array of q parameters
                 polynomial='Hermite',  # what type of polynomial to use
                 method='maxLike',  # The method, maximum likelihood or ...
                 poly_param=None,  # Some plynomials accept a parameter.
                 ):
        # I am changing the paradigm in this class. Different from the Matlab
        # implementation, this will not get a set of indexes for selecting the
        # training and the testing data. This will recieve the data directly
        # on the "fit" method and calculate everithing
        self.p = p
        self.q = q
        self.polynomial = polynomial
        self.method = method
        self.poly_param = poly_param
        # I have no idea if this is good practice, but I do
        # not like having attributes scattered around the code without being
        # defined in the constructor
        # This will return an array of unique solutions
    # What do I need?
    # 1. create an empty list of unique observables
    # 2. for each observalbe, calculate the regression, and return an array of
    # regression objects.
    # 3. return a dictionary whose indexes correspond to the incresing error
    # from the regression objects

    def fit(self, training_data):
        # takes the training data and returns itself with updated parameters
        # for readability, I am going to unpack things before calling the
        # function
        nsv = training_data[0]['sv'].shape[1]  # Number of state variables
        if 'u' in training_data[0]:
            nu = training_data[0]['u'].shape[1]  # number of inputs
        else:
            nu = 0

        obs_list = pqEDMD.observable_list(nsv,
                                          nu,
                                          self.polynomial,
                                          self.p,
                                          self.q)
        # That nique obs list was hard...
        # Coninue, I need to create the matrices for the evaluation of the
        # observalbes.
        # Return the scalers as an attribute
        xtr, ytr = pqEDMD.snapshots(training_data)
        # I have the data, next, perform the decomposition
        # Now, I should call the prefered decomposition for all the unique
        # observables in the observables list
        decompositions = [[] for _ in range(len(obs_list))]  # preallocation
        for decomposition in range(len(obs_list)):
            decompositions[decomposition] = getattr(dc,
                                                    f"{self.method}decomposition")(obs_list[decomposition], xtr, ytr)
        return decompositions

    @staticmethod
    def observable_list(nsv, nu, poly_type, p, q, param=None):
        # returns an array of unique observables to feed into the decomposition
        # iterate over p and q and create a list of observalbes.
        # To avoid a long list of if eflif elif eflif ... I am going to call an
        # isntance of the orthogonal polynomial subclass according to the
        # poly_type string.
        obs_inst = getattr(obs, f"{poly_type.lower()}Obs")
        # Do a set comprenhension to store unique elements and turn it
        # into a list
        obs_list = list({obs_inst(pq[0], pq[1], nsv, nu, param)
                         for pq in product(p, q)})

        return obs_list

    def snapshots(training_data):
        # Function to get the relevant structures for the calculation of the
        # observables. I whish I knew how to do the covariance a-la n4sid
        # So, if there is no testing dataset, I will use the same training set
        # for testing the algorithm
        # Summary of the matab code
        # 1. number of samples
        # 2. get the sequence untill the -2 index
        # 3. get the seuqnce from the 1 until the -1 index
        # 5. Normalize if it is the case
        x_prev = np.concatenate(  # concatenate all x in the training set
            # from the first until the antepenultimate
            [training_data[sample]['sv'][:-2, :]
             for sample in range(len(training_data))], axis=0
        )
        # concatenate all u in the training set
        # form the first until the antepenultimate
        if 'u' in training_data[0]:
            u_prev = np.concatenate([training_data[sample]['u'][:-2, :]
                                     for sample in range(len(training_data))],
                                    axis=0
                                    )
        x_post = np.concatenate([training_data[sample]['sv'][1:-1, :]
                                 for sample in range(len(training_data))],
                                axis=0
                                )
        if 'u' in training_data[0]:
            u_post = np.concatenate([training_data[sample]['u'][1:-1, :]
                                     for sample in range(len(training_data))],
                                    axis=0
                                    )
        #
        # I am going to return a normalizer or scaler for the state and for the
        # input in different objects, this make the calculation easier at later
        # stages
        # After somen tought, the scaling should be responsibility of the user,
        # before doing anything with the algorithm. But it is coded, so...
        if 'u' in training_data[0]:
            xtr = np.concatenate((x_prev, u_prev), axis=1)
            ytr = np.concatenate((x_post, u_post), axis=1)
        else:
            xtr = x_prev
            ytr = x_post
        return xtr, ytr

    @staticmethod
    def _xprev(x_prev, scalers):
        if scalers['xscaler'] is not None:
            xprev = list(scalers['xscaler'].transform(
                x_prev.reshape(1, -1)
            )[0])
        else:
            xprev = x_prev
        return xprev

    @staticmethod
    def _xprev_u(x_prev, u, scalers):
        if scalers['xscaler'] is not None:
            xprev = list(np.concatenate((
                scalers['xscaler'].transform(
                    x_prev.reshape(1, -1)
                )[0],
                scalers['uscaler'].transform(
                    u.reshape(1, -1)
                )[0])
            ))
        else:
            xprev = list(np.concatenate(
                (x_prev, u)))
        return xprev
