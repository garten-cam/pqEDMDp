'''
Author: Camilo Garcia Tenorio
Returns an observable object according to the type of polynomial
the type and the p, q parameters
'''
from sympy.polys import orthopolys as op
from sympy import symbols, Matrix, zeros, lambdify
import math
import numpy as np


class pqObservable:
    def __init__(self,
                 p=2,
                 q=0.8,
                 nSV=2,
                 nU=1,
                 huge_p_matrix=None):
        self.p = p
        self.q = q
        self.nSV = nSV
        self.nU = nU
        self.p_matrix = huge_p_matrix
        self.pq_matrix = None
        self.r_trx = None
        self.r_matrix = None

    @property
    def p_matrix(self):
        return self._p_matrix

    @p_matrix.setter
    def p_matrix(self, value):
        if value is not None:
            self._p_matrix = value
        elif (self.p**(self.nSV) - 1) > 9.0e+18:
            raise ValueError("number of state variables and p value\
                    combination exceeds maximum size")
        else:
            # preallocate the final matrix
            pm = np.zeros((self.nSV, (self.p + 1)**(self.nSV)), dtype=int)
            for col in range((self.p + 1)**(self.nSV)):
                base_string = np.base_repr(col, self.p + 1)
                base_string = '0' * (self.nSV - len(base_string)) + base_string
                pm[:, col] = np.flip([int(x) for x in base_string])
        # For the qr or orthogonalization parts in the following,
        # it is better to have the poynomials ordered.
        # Assume the lexicographical ordering according to the
        # state variables. i.e., x_1 < x_2 if ||x_1|| < ||x_2||
        # and, if the norm of two arbitrary polynomials is equal,
        # the lowest component determines the ordering
        # [0,1,1]'<[1,1,0]'
        self._p_matrix = pm[:, np.argsort(np.linalg.norm(pm, axis=0, ord=self.p),
                                          kind='stable')]
        # Add the effect of u

    @property
    def pq_matrix(self):
        return self._pq_matrix

    @pq_matrix.setter
    def pq_matrix(self, value):
        if value is not None:
            self._pq_matrix = value
        else:
            # trim according to q
            pqm = self.p_matrix[:, np.linalg.norm(self.p_matrix,
                                                  ord=self.q,
                                                  axis=0) <= self.p]
            # add the rows and columns of the input
            self._pq_matrix =\
                np.concatenate(
                    (np.concatenate((pqm,
                                     np.zeros([self.nSV, self.nU])),
                                    axis=1),
                     np.concatenate((np.zeros([self.nU, pqm.shape[1]]),
                                     np.eye(self.nU)), axis=1)),
                    axis=0)

    @property
    def r_trx(self):
        return self._r_trx

    @r_trx.setter
    def r_trx(self, value):
        # This is to use orthogonalization from the data.
        # If it is not provided,
        # Just use an identity matrix for weighting the observables.
        pq_m_rows = self.pq_matrix.shape[1]
        if value is not None:
            # check if the matrix is the tright size
            if value.shape == (pq_m_rows, pq_m_rows):
                self._r_trx = value
        else:
            self._r_trx = np.identity(pq_m_rows)

    @property
    def r_matrix(self):
        return self._r_matrix

    @r_matrix.setter
    def r_matrix(self, value):
        # This is to use orthogonalization from the data.
        # If it is not provided,
        # Just use an identity matrix for weighting the observables.
        pq_m_rows = self.pq_matrix.shape[1]
        if value is not None:
            # check if the matrix is the right size
            if value.shape == (pq_m_rows, pq_m_rows):
                self._r_matrix = value
        else:
            self._r_matrix = np.identity(pq_m_rows)

    @property
    def pq_function(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        # Add otrhogonalization, just in case there is a r_trx different than
        # identity. Don't do it as an if... just do the multiplication
        poly_prod = Matrix(np.matmul(self.poly_prod, self.r_trx))
        # lambdify...
        return lambdify(x, poly_prod[1:], modules='numpy')

    def __eq__(self, other):
        equal = False
        # If the shapes are not equal, they are definitely not equal
        if self.pq_matrix.shape == other.pq_matrix.shape:
            # If the shapes are equal then... Compare the whole array
            equal = (self.pq_matrix == other.pq_matrix).all()
        return equal

    def __hash__(self):
        return hash((self.p))


class hermiteObs(pqObservable):
    @property
    def poly_prod(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store
        # the polynomials
        polys = zeros(*indexes.shape)
        if self.nU > 0:
            for idx, sv in enumerate(x[:-self.nU]):
                polys[idx, :] = Matrix([op.hermite_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
            for idx, u in enumerate(x[-self.nU:]):
                # Legendre polys in the first order are the constant function
                # l^1(x)=x, perfect for the inputs
                polys[idx + self.nSV, :] =\
                    Matrix([op.legendre_poly(power, u)
                            for power in indexes[idx + self.nSV, :]]).transpose()
        else:
            for idx, sv in enumerate(x):
                polys[idx, :] = Matrix([op.hermite_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
        # take the product per column
        return Matrix([math.prod(polys[:, col])
                       for col in range(polys.shape[1])]).transpose()


class laguerreObs(pqObservable):
    @property
    def poly_prod(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store
        # the polynomials
        polys = zeros(*indexes.shape)
        if self.nU > 0:
            for idx, sv in enumerate(x[:-self.nU]):
                polys[idx, :] = Matrix([op.laguerre_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
            for idx, u in enumerate(x[-self.nU:]):
                # Legendre polys in the first order are the constant function
                # l^1(x)=x, perfect for the inputs
                polys[idx + self.nSV, :] =\
                    Matrix([op.legendre_poly(power, u)
                            for power in indexes[idx + self.nSV, :]]).transpose()
        else:
            for idx, sv in enumerate(x):
                polys[idx, :] = Matrix([op.laguerre_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
        # take the product per column
        return Matrix([math.prod(polys[:, col])
                       for col in range(polys.shape[1])]).transpose()


class chebyshevtObs(pqObservable):
    @property
    def poly_prod(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store
        # the polynomials
        polys = zeros(*indexes.shape)
        if self.nU > 0:
            for idx, sv in enumerate(x[:-self.nU]):
                polys[idx, :] = Matrix([op.chebyshevt_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
            for idx, u in enumerate(x[-self.nU:]):
                # Legendre polys in the first order are the constant function
                # l^1(x)=x, perfect for the inputs
                polys[idx + self.nSV, :] =\
                    Matrix([op.legendre_poly(power, u)
                            for power in indexes[idx + self.nSV, :]]).transpose()
        else:
            for idx, sv in enumerate(x):
                polys[idx, :] = Matrix([op.chebyshevt_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
        # take the product per column
        return Matrix([math.prod(polys[:, col])
                       for col in range(polys.shape[1])]).transpose()


class chebyshevuObs(pqObservable):
    @property
    def poly_prod(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store
        # the polynomials
        polys = zeros(*indexes.shape)
        if self.nU > 0:
            for idx, sv in enumerate(x[:-self.nU]):
                polys[idx, :] = Matrix([op.chebyshevu_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
            for idx, u in enumerate(x[-self.nU:]):
                # Legendre polys in the first order are the constant function
                # l^1(x)=x, perfect for the inputs
                polys[idx + self.nSV, :] =\
                    Matrix([op.legendre_poly(power, u)
                            for power in indexes[idx + self.nSV, :]]).transpose()
        else:
            for idx, sv in enumerate(x):
                polys[idx, :] = Matrix([op.chebyshevu_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
        # take the product per column
        return Matrix([math.prod(polys[:, col])
                       for col in range(polys.shape[1])]).transpose()


class legendreObs(pqObservable):
    @property
    def poly_prod(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store
        # the polynomials
        polys = zeros(*indexes.shape)
        if self.nU > 0:
            for idx, sv in enumerate(x[:-self.nU]):
                polys[idx, :] = Matrix([op.legendre_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()
            for idx, u in enumerate(x[-self.nU:]):
                # Legendre polys in the first order are the constant function
                # l^1(x)=x, perfect for the inputs
                polys[idx + self.nSV, :] =\
                    Matrix([op.legendre_poly(power, u)
                            for power in indexes[idx + self.nSV, :]]).transpose()
        else:
            for idx, sv in enumerate(x):
                polys[idx, :] = Matrix([op.legendre_poly(power, sv)
                                        for power in indexes[idx, :]]).transpose()

        # take the product per column
        return Matrix([math.prod(polys[:, col])
                       for col in range(polys.shape[1])]).transpose()


if __name__ == "__main__":
    nSV = 3
    nU = 0
    obs = hermiteObs(nSV=nSV, p=2, q=0.9, nU=nU)
    obs_fun = obs.pq_function
    xx = obs_fun(3, 5, 6)
    print(xx)
    rng = np.random.default_rng()
    obs.r_trx = np.triu(rng.random((7, 7)))
    obs_fun = obs.pq_function
    xx = obs_fun(3, 5, 6)
    print(xx)
