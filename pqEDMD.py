# pqEDMD Implementation
'''
Author: Camilo Garcia Tenorio
Development of pqEDMD in python... lets see how this goes...
'''
import pqObservable as obs
import decompositions as dc
from itertools import product
import numpy as np
from sklearn import preprocessing


class pqEDMD:
    def __init__(self,
                 p=2,  # it can be an array of p parameters
                 q=0.9,  # it can be an array of q parameters
                 polynomial='Hermite',  # what type of polynomial to use
                 method='maxLike',  # The method, maximum likelihood or ...
                 poly_param=None,  # Some plynomials accept a parameter.
                 normalization=True,
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
        self.normalization = normalization
        self.xscaler = None  # I have no idea if this is good practice, but I do 
        # not like having attributes scattered around the code without being
        # defined in the constructor
        self.uscaler = None
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
        nu = training_data[0]['u'].shape[1]  # number of inputs
        obs_list = pqEDMD.observable_list(nsv,
                                          nu,
                                          self.polynomial,
                                          self.p,
                                          self.q)
        # Thatunique obs list was hard...
        # Coninue, I need to create the matrices for the evaluation of the 
        # observalbes.
        # Return the scalers as an attribute
        xtr, ytr, self.xscaler, self.uscaler = pqEDMD.snapshots(training_data)
        # I have the data, next, perform the decomposition
        # Now, I should call the prefered decomposition for all the unique 
        # observables in the observables list
        decp = getattr(dc, f"{self.method}Decomp")
        # I am completeky lost in this bs... ok, I need to call a decomposition,
        # that recieves an observable as an argument. The problem is that the
        # function in the observable is not vectorized. Keep it that way. 
        # TODO [ ] get the vectorized function
        
        return self

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

    def snapshots(training_data, normalization=True):
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
            [training_data[sample]['sv'][:-2,:]
             for sample in range(len(training_data))],axis=0
        )
        u_prev = np.concatenate(  # concatenate all u in the training set
            # form the first until the antepenultimate
            [training_data[sample]['u'][:-2,:]
             for sample in range(len(training_data))],axis=0
        )
        x_post = np.concatenate(
            [training_data[sample]['sv'][1:-1,:]
             for sample in range(len(training_data))],axis=0
        )
        u_post = np.concatenate(
            [training_data[sample]['u'][1:-1,:]
             for sample in range(len(training_data))],axis=0
        )
        # 
        # I am going to return a normalizer or scaler for the state and for the 
        # input in different objects, this make the calculation easier at later
        # stages
        if normalization:
            xscaler = preprocessing.StandardScaler()
            uscaler = preprocessing.StandardScaler()
            xtr = np.concatenate(
                (xscaler.fit_transform(x_prev),
                 uscaler.fit_transform(u_prev)), axis=1
            )
            ytr = np.concatenate(
                (xscaler.transform(x_post), uscaler.transform(u_post)), axis=1
            )
        else:
            xtr = np.concatenate((x_prev, u_prev), axis=1)        
            xscaler = None
            ytr = np.concatenate((x_post, u_post), axis=1)
            uscaler = None
        return xtr, ytr, xscaler, uscaler
        