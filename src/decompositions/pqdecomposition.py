"""
Author: Camilo Garcia Tenotio
returns a pq decomposition object
"""

import sys
import numpy as np
from sympy import Matrix, linear_eq_to_matrix, linsolve, symbols
import matplotlib.pyplot as plt

from pqobservable import pqObservable


class pqDecomposition:
    """
    Class for performing a decomposition to a particular system based on a set
    of observables.

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

    def __init__(
        self,
        observable: pqObservable,
        system: dict,
    ):
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

        self.num_obs: int = np.shape(self.observable.pq_mat())[1]
        self.sys_m: int = n_in
        self.sys_l: int = self.observable.obs_l
        self.sys_n: int = self.num_obs + 1
        # Calculate
        U = self.regression(o_fut, o_pst)
        # Get the system matrices
        self.A: np.ndarray = self.matrix_A(U)
        self.B: np.ndarray = self.matrix_B(U)
        self.C: np.ndarray = self.matrix_C()
        self.D: np.ndarray = np.zeros((self.sys_l, np.maximum(1, self.sys_m)))

    def regression(self, lhs, rhs):
        '''
        Solves lhs = sol * rhs
        '''
        g = rhs.T@rhs
        a = rhs.T@lhs
        return np.linalg.inv(g)@a

    def matrix_A(self, u) -> np.ndarray:
        if self.sys_m == 0:
            a = u.T
        else:
            a = u[:-self.sys_m, :-self.sys_m].T

        return a

    def matrix_B(self, u):
        if self.sys_m:
            b = u[-self.sys_m:, :-self.sys_m].T
        else:
            b = np.zeros((self.sys_n, 1))
        return b

    def matrix_C(self):
        # this one is complicated, I must do all the inverse of the
        # transformations.
        # Given the ordering of the polynomials, we need the additive inverse
        # of the first "l" terms
        orders = self.observable.poly_base()
        order_one_o = Matrix(orders[:self.sys_l])
        # we also need dummy variables for representing the functions
        # As many dummy variables as there are states in the system
        z = Matrix([*symbols(f"z:{self.sys_l}")])
        # define again the variables
        x = symbols(f"x:{self.sys_l}")
        # Equation system to solve
        f_to_inv = order_one_o - z
        # Convert to a system of equations
        A, b = linear_eq_to_matrix(f_to_inv, x)
        # solve the system
        sol = linsolve((A, b), x)
        # get the new matrices for A
        Az, _ = linear_eq_to_matrix(
            sol.args[0], [symbols("1"), *symbols(f"z:{self.sys_l}")]
        )
        # Add the constant term in case b = constant +- z
        # Step by Step
        ct_sol = linsolve(b, symbols(f"z:{self.sys_l}"))
        Az[:, 0] = Matrix(ct_sol.args[0])
        # create the c matrix
        c = np.zeros(
            (self.observable.obs_l, self.observable.pq_mat().shape[1] + 1))
        # Assign the c entries
        c[:, : self.sys_l + 1] = Az
        return c

    def y_snapshots(self, system) -> np.ndarray:
        obsrv = self.observable.obs_fun()
        # Evaluate the Ys. In a system there are many samples. This function
        # takes the outputs "y" of the system and,
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

    def u_snapshots(self, system) -> np.ndarray:
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
        error = np.sum(errors) / self.observable.obs_l / len(system)
        return error

    def abs_error(self, system) -> np.ndarray:
        predictions = self.predict_from_test(system)
        ab_err = [
            np.sum(
                np.absolute(pred["y"] - sys_i["y"])
            ) / pred["y"].shape[0]
            for pred, sys_i in zip(predictions, system)
        ]
        return np.sum(ab_err) / self.observable.obs_l / len(system)

    def predict_from_test(self, system):
        # Get the initial conditions
        y0 = [sys_i["y"][0, :] for sys_i in system]
        n_p = [np.shape(sys_i["t"])[0] for sys_i in system]
        if self.sys_m != 0:
            pred_test = self.predict(y0, n_p, [sys_i["u"] for sys_i in system])
        else:
            pred_test = self.predict(y0, n_p)
        return pred_test

    def predict(self, y0, n_points, u=[]):
        # if the input is none, assign it as an array of zeros
        # Save the obse:vable to avoid verbosity
        obs = self.observable.obs_fun()
        if self.sys_m == 0:
            u = [np.zeros((n_pi, 1)) for n_pi in n_points]

        # I think preallocation may help in doing this
        prediction = [
            {'y': np.zeros((n_p, self.observable.obs_l))} for n_p in n_points
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
        plt.ion()
        plt.figure()
        e = np.linalg.eig(self.A)
        plt.scatter(np.real(e.eigenvalues), np.imag(e.eigenvalues))
        t = np.linspace(0, 2 * np.pi, 600)
        plt.plot(np.sin(t), np.cos(t))
