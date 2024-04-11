"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
"""

import sys

import numpy as np
from sympy import Matrix, linear_eq_to_matrix, linsolve, symbols
import matplotlib.pyplot as plt


class pqDecomposition:

    def __init__(
        self,
        observable,
        system,
    ):
        # General Decomposition class.
        # Implements the ordinary squa
        # es solution
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
        self.l = self.observable.l
        self.n = np.shape(self.observable.pq_mat())[1] + 1
        # Get the system matrices
        self.A = self.matrix_A(U)
        self.B = self.matrix_B(U)
        self.C = self.matrix_C()
        self.D = np.zeros((self.l, self.m))

    def matrix_A(self, u):
        return u[:-self.m, :-self.m].T

    def matrix_B(self, u):
        if self.m:
            b = u[-self.m:, :-self.m].T
        else:
            b = np.zeros((self.n, 1))
        return b

    def matrix_C(self):
        # this one is complicated, I must do all the inverse of the
        # transformations.
        # Given the ordering of the polynomials, we need the additive inverse
        # of the first "l" terms
        orders = self.observable.poly_base()
        order_one_o = Matrix(orders[:self.l])
        # we also need a dummy variable for the representing the functions
        # As many dummy variables as there are states in the system
        z = Matrix([*symbols(f"z:{self.l}")])
        # define again the variables
        x = symbols(f"x:{self.l}")
        # Equation system to solve
        f_to_inv = order_one_o - z
        # Convert to a system of equations
        A, b = linear_eq_to_matrix(f_to_inv, x)
        # solve the system
        sol = linsolve((A, b), x)
        # get the new matrices for A
        Az, _ = linear_eq_to_matrix(
            sol.args[0], [symbols("1"), *symbols(f"z:{self.l}")]
        )
        # Add the constant term in case b = constant +- z
        # Step by Step
        ct_sol = linsolve(b, symbols(f"z:{self.l}"))
        Az[:, 0] = Matrix(ct_sol.args[0])
        # create the c matrix
        c = np.zeros(
            (self.observable.l, self.observable.pq_mat().shape[1] + 1))
        # Assign the c entries
        c[:, : self.l + 1] = Az
        return c

    def y_snapshots(self, system):
        # function to evaluate the polynomials
        obsrv = self.observable.obs_fun()
        # Evaluate the Ys. In a system there are many samples. This function
        # takes the outputs "y" of the system and:
        # applies the function while slicing the past and future samples
        y_ob_pst = np.vstack((
            [
                np.hstack((
                    np.ones((
                        np.shape(sp["y"])[0] - 2, 1
                    )),
                    np.squeeze(obsrv(*sp["y"][:-2, :].T).T)
                )) for sp in system
            ]
        ))
        y_ob_fut = np.vstack((
            [
                np.hstack((
                    np.ones((
                        np.shape(sp["y"])[0] - 2, 1
                    )),
                    np.squeeze(obsrv(*sp["y"][1:-1, :].T).T)
                )) for sp in system
            ]
        ))
        return y_ob_pst, y_ob_fut

    def u_snapshots(self, system):
        # This the same as the variables y without observation
        u_pst = np.concatenate([sp["u"][:-2, :] for sp in system], axis=0)
        u_fut = np.concatenate([sp["u"][1:-1, :] for sp in system], axis=0)
        return u_pst, u_fut

    def error(self, system):
        predictions = self.predict_from_test(system)
        # calculate the error from all the predictions
        errors = [
            np.sum(
                np.absolute(pred["y"] - sys_i["y"]) /
                (np.absolute(sys_i["y"]) + sys.float_info.epsilon)
            ) / pred["y"].shape[0]
            for pred, sys_i in zip(predictions, system)
        ]
        error = np.sum(errors) / self.observable.l / len(system)
        return error

    def predict_from_test(self, system):
        # Get the initial conditions
        y0 = [sys_i["y"][0, :] for sys_i in system]
        n_p = [np.shape(sys_i["t"])[0] for sys_i in system]
        u = [sys_i["u"] for sys_i in system]

        pred_test = self.predict(y0, n_p, u)
        return pred_test

    def predict(self, y0, n_points, u=[]):
        # if the input is none, assign it as an array of zeros
        # Save the obse:vable to avoid verbosity
        obs = self.observable.obs_fun()
        # I think preallocation may help in doing this
        prediction = [
            {'y': np.zeros((n_p, self.observable.l))} for n_p in n_points
        ]
        for o_ind, orbit in enumerate(prediction):
            orbit['y'][0, :] = y0[o_ind]
            for s_ind, _ in enumerate(orbit["y"][1:, :]):
                x_prev = orbit["y"][s_ind, :]
                x_post = self.A @ np.hstack((
                    np.ones((1, 1)),
                    obs(*x_prev).T
                )).T + self.B @ u[o_ind][s_ind, :].reshape(1, -1).T
                orbit['y'][s_ind + 1, :] = (self.C @ x_post +
                                            self.D @ u[o_ind][s_ind, :].reshape(1, -1).T).T
        return prediction

    def spectrum(self):
        plt.figure()
        e = np.linalg.eig(self.A)
        plt.scatter(np.real(e.eigenvalues), np.imag(e.eigenvalues))
        t = np.linspace(0, 2 * np.pi, 600)
        plt.plot(np.sin(t), np.cos(t))
