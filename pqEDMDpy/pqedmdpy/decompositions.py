'''
Author: Camilo Garcia Tenorio
returns a decomposition object
'''
import numpy as np
import sys
from scipy.optimize import minimize
from functools import partial
from sympy import symbols, linear_eq_to_matrix, Matrix, linsolve, lambdify


class decomposition(object):
    def __init__(
            self,
            observable,
            xtr,
            ytr,
    ):
        # Generic class for defining a decomposition
        self.observable = observable
        self._U = self._set_U(xtr, ytr)

    @property
    def U(self):
        return self._U

    def _set_U(self, xtr, ytr):
        g = self.g(xtr)
        a = self.a(xtr, ytr)
        # with the g and a matrices, get the transpose of u
        return np.matmul(np.linalg.inv(g), a)

    @property
    def A(self):
        # Extrat the A matrix from U; x=Ax
        if not self.observable.nU:
            a = np.transpose(self.U)
        else:
            a = np.transpose(self.U[:-self.observable.nU - 1,
                                    :-self.observable.nU - 1])
        return a

    @property
    def B(self):
        # extract the B matrix from U; x=Ax+Bu
        if not self.observable.nU:
            b = None
        else:
            b = self.U[-self.observable.nU:, :-self.observable.nU - 1]
        return b

    @property
    def C(self):
        # this one is complicated, I must do all the inverse of the
        # transformations.
        # x = QR, ro recover R because all the data lives in Q now
        # R = np.linalg.inv(self.observable.r_trx)
        # Given the ordering of the polynomials, we need the additive inverse
        # of the first nSV terms
        order_one_obs = Matrix(
            self.observable.poly_prod[:self.observable.nSV + 1])
        # order_one_obs[0] = symbols('1')
        # we also need a dummy variable for the representing the functions
        # As many dummy variables as there are states in the system
        z = Matrix([symbols('1'), *symbols(f'z:{self.observable.nSV}')])
        # define again the variables
        x = symbols(f'x:{self.observable.nSV}')
        # Equation system to solve
        f_to_inv = order_one_obs - z
        # Convert to a system of equations
        A, b = linear_eq_to_matrix(list(f_to_inv), [symbols('1'), *x])
        # solve the system
        sol = linsolve((A, b), [symbols('1'), *x])
        # get the new matrices for A and b
        Az, _ = linear_eq_to_matrix(
            list(sol)[0],
            [symbols('1'), *symbols(f'z:{self.observable.nSV}')])
        # Remove the constant term
        Az = Az[1:, :]
        # Add the constant term in case b = number +- z
        Az[:, 0] = Matrix(
            list(linsolve((b[1:]),
                          symbols(f'z:{self.observable.nSV}')))).T
        # create the c matrix
        c = np.zeros((self.observable.nSV,
                      self.observable.pq_matrix.shape[1]))
        # Assign the c entries
        c[:, :self.observable.nSV + 1] = Az
        return c

    def g(self, xtr):
        # I am going to divide the "g" and "a" matrix calcualtion.
        # That way, I can apply all the necessary calculations on "g"
        # and the observables. For orthogonalization and polynomial
        # manipulation purposes
        # Get the observables function
        obs = self.observable
        # Evaluate the training data in x and y
        x_eval = self.xy_eval(obs, xtr)
        # With the QR decomposition I can orthogonalize the evaluations and
        # get the well conditioned g matrix.
        # Get the orthogonal x_eval
        obs.r_matrix = np.linalg.qr(x_eval)[1]  # gets the second output
        # # Assign the orthogonalization matrix
        obs.r_trx = np.linalg.inv(obs.r_matrix)
        # # evaluate again
        x_eval = self.xy_eval(obs, xtr)
        g = np.matmul(x_eval.transpose(), x_eval)/x_eval.shape[0]
        # It is a good idea to orthogonalize here, always.

        return g

    def a(self, xtr, ytr):
        # The same as g but with a: x'y
        obs = self.observable
        x_eval = self.xy_eval(obs, xtr)
        y_eval = self.xy_eval(obs, ytr)
        a = np.matmul(x_eval.transpose(), y_eval)/x_eval.shape[0]
        return a

    @staticmethod
    def is_invertible(A):
        return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]

    @staticmethod
    def xy_eval(obs, xtr):
        psi = obs.pq_function
        return np.concatenate((np.ones((xtr.shape[0], 1))*obs.r_trx[0][0],
                               np.array(psi(*xtr.transpose())).transpose()),
                              axis=1)

    @property
    def evol_function(self):
        x = symbols(f'x:{self.observable.nSV + self.observable.nU}')
        # Add otrhogonalization, just in case there is a r_trx different than
        # identity. Don't do it as an if... just do the multiplication
        poly_prod = Matrix(np.matmul(np.matmul(self.observable.poly_prod,
                                     self.observable.r_trx),
                                     self.U))
        # lambdify...
        return lambdify(x, poly_prod, modules='numpy')

    def predict(self, x0, n_points, u=None):
        ev_fun = self.evol_function
        # preallocate
        pred = [{'sv': np.zeros((n_points[i], self.observable.nSV))}
                for i in range(len(x0))]
        # populate the initial condition
        for sample in range(len(x0)):
            pred[sample]['sv'][0, :] = x0[sample]
        # main loop to assign all values
        for sample in range(len(x0)):
            for step in range(1, n_points[sample]):
                if u is None:
                    xprev = pred[sample]['sv'][step-1, :]
                else:
                    xprev = np.concatenate((pred[sample]['sv'][step-1, :],
                                           u[sample][step-1, :]))

                # evolve
                xpost = ev_fun(*xprev)
                # in case of orthogonalization, bring it back to the original
                # observables
                xpost_ogn = xpost @ self.observable.r_matrix
                # bring it to the original space state
                pred[sample]['sv'][step, :] = (self.C @ xpost_ogn.T).T
        return pred

    def pred_from_test(self, ts):
        # to make it easier in error calculation, this method take a subset of
        # the data in standard form and returns a prediction.
        x0 = [ts[i]['sv'][0, :] for i in range(len(ts))]
        # points per orbit
        np = [ts[i]['sv'].shape[0] for i in range(len(ts))]
        if 'u' in ts[0]:
            pred = self.predict(x0, np, [ts[i]['u'] for i in range(len(ts))])
        else:
            pred = self.predict(x0, np)
        return pred

    def error(self, ts):
        # predict the trajectories
        pred = self.pred_from_test(ts)
        # Extract all the sv's from the test set
        ts_sv = np.concatenate([ts[i]['sv'][1:, :] for i in range(len(ts))])
        # Extract the sv's from the pred
        pr_sv = np.concatenate([pred[i]['sv'][1:, :]
                               for i in range(len(pred))])
        return np.sum(np.abs(ts_sv - pr_sv))/len(ts_sv)/self.observable.nSV


class maxLikedecomposition(decomposition):
    def __init__(self, observable, xtr, ytr):
        # super().__init__(observable, xtr, ytr)
        self.observable = observable
        self._U0 = self._set_U0(xtr, ytr)
        self._Q = self._set_Q(xtr, ytr)
        self._U = self._set_U(xtr, ytr)

    @property
    def U0(self):
        return self._U0

    def _set_U0(self, xtr, ytr):
        return decomposition(self.observable, xtr, ytr).U

    @property
    def Q(self):
        return self._Q

    def _set_Q(self, xtr, ytr):
        obs = self.observable  # save in its own variable, makes the
        # code more readable
        # evaluate the polybase
        x_eval = self.xy_eval(obs, xtr)
        # Covariance calculation
        # mean value per column
        x_mean = np.mean(x_eval, axis=0)
        # q matrix
        return np.matmul((x_eval - x_mean).transpose(), (x_eval - x_mean)) /\
            (x_eval.shape[0] - 1)

    @property
    def U(self):
        return self._U

    def _set_U(self, xtr, ytr):
        # save the observable function. Not necesssary, but ok...
        obs = self.observable  # save in its own variable, makes the
        # code more readable
        # evaluate the polybase
        x_eval = self.xy_eval(obs, xtr)
        y_eval = self.xy_eval(obs, ytr)
        # Now we are ready for the optimization
        # number of observables to fit
        n_obs = self.observable.pq_matrix.shape[1]
        u = np.identity(self._U0.shape[0])  # Preallocate
        for u_column in range(1, n_obs):
            # turn the objective function into a new function dependent only on
            # one parameter
            obj_f = partial(maxLikedecomposition.cost_function,
                            Q=self.Q,
                            sigma=self.Q[u_column, u_column],
                            x_eval=x_eval,
                            y_eval=y_eval[:, u_column])
            u[:, u_column] = minimize(obj_f,  # New objective function f(u_col)
                                      self._U0[:, u_column] - \
                                      1,  # Initial condition
                                      method='BFGS',
                                      options={'disp': False}).x

        return u

    @staticmethod
    def cost_function(u_col, Q, sigma, x_eval, y_eval):
        j = np.sum((
            y_eval - np.array(
                np.matmul(x_eval,
                          u_col))
        )**2/(sigma**2 +
              np.matmul(np.matmul(u_col, Q),
                        np.atleast_2d(u_col).T)))
        return j


class rrrdecomposition(decomposition):
    def __init__(self, observable, xtr, ytr):
        self.observable = observable
        self._U = self._set_U(xtr, ytr)

    @property
    def U(self):
        return self._U

    def _set_U(self, xtr, ytr):
        x_eval = self.xy_eval(self.observable, xtr)
        y_eval = self.xy_eval(self.observable, ytr)
        # Calculate the singular value decomposition
        U, S, V = np.linalg.svd(x_eval)
        # calculatethe effective rank of xeval
        r = 1
        n_obs = self.observable.pq_matrix.shape[1]
        while ((r < n_obs) and
               (S[r] > max(x_eval.shape)*sys.float_info.epsilon*S[0])):
            r += 1

        d = U.T @ y_eval
        d_rSlice = d[:r, :]
        d_sWeight = (d_rSlice.T*(1/S[:r])).T
        d_nobsSlice = np.concatenate(
            (d_sWeight, np.zeros((x_eval.shape[1] - r, n_obs))))
        u = V.T @ d_nobsSlice
        return u
