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
                 normalization=False,
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
        # I have no idea if this is good practice, but I do
        self.scalers = {'xscaler': None}
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
        xtr, ytr, self.scalers = pqEDMD.snapshots(training_data,
                                                  self.normalization)
        # I have the data, next, perform the decomposition
        # Now, I should call the prefered decomposition for all the unique
        # observables in the observables list
        decompositions = [[] for _ in range(len(obs_list))]  # preallocation
        for decomposition in range(len(obs_list)):
            decompositions[decomposition] = getattr(dc,
                                                    f"{self.method}Decomp")(obs_list[decomposition], xtr, ytr)
        return decompositions

    @staticmethod
    def predict(decp, x0, scalers, n_points, u=None):
        # ok, do it for one and then set the for
        # Get the observable that includes the constant term
        ev_fun = decp.evol_function
        # preallocate
        pred = [{'sv': np.zeros((n_points, decp.observable.nSV))}
                for _ in range(len(x0))]
        # There are two options, if there is an input or not.
        # I am creating two sub functions to avoid a huge if-else
        # assign the initial condition
        for sample in range(len(x0)):
            pred[sample]['sv'][0, :] = x0[sample]
        # Main loop to assign all the values
        for sample in range(len(x0)):
            for step in range(1, n_points):
                # call the predictions here
                if u is None:
                    xprev = pqEDMD._xprev(pred[sample]['sv'][step-1, :],
                                          scalers)
                else:
                    xprev = pqEDMD._xprev_u(pred[sample]['sv'][step-1, :],
                                            u[sample][step-1, :],
                                            scalers)

                # apply the necessary tranformations to xprev
                # 1. evolve
                xpost = ev_fun(*xprev)
                # 2. If there was orthogonalization, bring back
                # to the original obs
                xpost_ogn = np.matmul(xpost, decp.observable.r_matrix)
                # 3. Bring now to the original state space
                xss = np.transpose(np.matmul(decp.C, np.transpose(xpost_ogn)))
                # Do not like the double transpose but it is consistent with
                # the lit
                # 4. descale if necessary
                if scalers['xscaler'] is not None:
                    pred[sample]['sv'][step,
                                       :] = scalers['xscaler'].inverse_transform(xss)
                else:
                    pred[sample]['sv'][step, :] = xss
        return pred

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
        if normalization:
            scalers = {'xscaler': preprocessing.MinMaxScaler()}
            if 'u' in training_data[0]:
                scalers['uscaler'] = preprocessing.MinMaxScaler()
                xtr = np.concatenate(
                    (scalers['xscaler'].fit_transform(x_prev),
                     scalers['uscaler'].fit_transform(u_prev)), axis=1)
                ytr = np.concatenate(
                    (scalers['xscaler'].transform(x_post),
                     scalers['uscaler'].transform(u_post)), axis=1)
            else:
                xtr = scalers['xscaler'].fit_transform(x_prev)
                ytr = scalers['xscaler'].fit_transform(x_post)

        else:
            if 'u' in training_data[0]:
                xtr = np.concatenate((x_prev, u_prev), axis=1)
                ytr = np.concatenate((x_post, u_post), axis=1)
            else:
                xtr = x_prev
                ytr = x_post
            scalers = {'xscaler': None}
            scalers['uscaler'] = None
        return xtr, ytr, scalers

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
            xprev = list(np.concatenate(
                scalers['xscaler'].transform(
                    x_prev.reshape(1, -1),
                    scalers['uscaler'].transform(
                        u
                    ))
            )[0])
        else:
            xprev = list(np.concatenate(
                (x_prev, u)))
        return xprev


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    # Test the implementation with the toy duffing equation
    def duffode(x, t):
        # Duffing with two AS points
        return [x[1], -0.5*x[1] + x[0] - x[0]**3]

    # rng = np.random.default_rng(1342)
    rng = np.random.default_rng(1342)
    num_ics = 10
    ics_width = 10
    ics = ics_width*rng.random((num_ics, 2)) - ics_width/2
    # ics = np.array([[-0.3319, -1.2550],
    #                  [0.8813, -0.6178],
    #                  [-1.9995, -0.4129],
    #                  [-0.7907, 0.1553],
    #                  [-1.4130, -0.3232],
    #                  [-1.6306, 0.7409]])

    t_end = 30
    n_points = 301

    # I need a dictionary with the necessary data.
    # samples = [{'sv':np.empty((n_points,2)), 't':np.empty((n_points,1))}\
    #             for _ in range(num_ics)]
    #
    # for sample in range(num_ics):
    #     t = np.linspace(0,t_end,n_points)
    #     sol = odeint(duffode, ics[sample, :], t)
    #     samples[sample]['sv'] = sol
    #     samples[sample]['t'] = t
    #     # samples[sample]['u'] = np.ones((n_points,1))*u[sample]
    #     # plt.plot(sol[:,0],sol[:,1])
    #     # plt.title(f'sample:{sample}')
    #
    #
    duff_EDMD = pqEDMD(p=[3], q=[1],
                       polynomial='Legendre', normalization=False)
    # # fit the alg
    tr = range(0, 2)
    ts = range(2, num_ics)
    # duff_decomps = duff_EDMD.fit([samples[i] for i in tr])
    # # duff_decomps = duff_EDMD.fit(samples[0])
    # # test with some samples
    # approximations = duff_EDMD.predict(duff_decomps[0],
    #                     [samples[i]['sv'][0,:] for i in ts],
    #                     duff_EDMD.scalers,
    #                     n_points)

    # plt.figure()
    # [plt.plot(samples[i]['sv'][:,0],samples[i]['sv'][:,1], 'r') for i in tr]
    # [plt.plot(samples[i]['sv'][:,0],samples[i]['sv'][:,1], 'b') for i in ts]
    # [plt.plot(approximations[i]['sv'][:,0],approximations[i]['sv'][:,1], 'k-.') for i in range(len(ts))]
    # plt.show()

    # Test the code for inputs
    # use some random input
    inputs = 100*rng.random((num_ics, 1)) - 50
    # The inputs will be a step that is also a random number

    def duffode_u(x, t, u):
        # Duffing with two AS points
        return [x[1], -0.5*x[1] + x[0] - x[0]**3 + u]

    # preallocate the samples list
    samples_u = [{'sv': np.empty((n_points, 2)),
                  't': np.empty((n_points, 1)), 'u': np.empty((n_points, 1))}
                 for _ in range(num_ics)]

    # populate the samples
    for sample in range(num_ics):
        t = np.linspace(0, t_end, n_points)
        sol = odeint(duffode_u, ics[sample, :], t, args=(inputs[sample][0],))
        samples_u[sample]['sv'] = sol
        samples_u[sample]['t'] = t
        samples_u[sample]['u'] = np.full((n_points, 1), inputs[sample][0])

    duff_decomps_u = duff_EDMD.fit([samples_u[i] for i in tr])

    app_u = duff_EDMD.predict(duff_decomps_u[0],
                              [samples_u[i]['sv'][0, :] for i in ts],
                              duff_EDMD.scalers,
                              n_points,
                              [samples_u[i]['u'] for i in ts])

    [plt.plot(samples_u[i]['sv'][:, 0], samples_u[i]['sv'][:, 1], 'r')
     for i in tr]
    [plt.plot(samples_u[i]['sv'][:, 0], samples_u[i]['sv'][:, 1], 'b')
     for i in ts]
    [plt.plot(app_u[i]['sv'][:, 0], app_u[i]
              ['sv'][:, 1], 'k-.') for i in range(len(ts))]
    plt.show()
    x = 1
