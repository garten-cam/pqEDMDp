'''
Author: Camilo Garcia Tenorio
Returns an observable object according to the type of polynomial, 
the type and the p, q parameters
'''
# Lets start with a simple script and build from there,
from sympy.polys import orthopolys as op
from sympy import symbols, Matrix, zeros, lambdify
import math
import numpy as np
class pqObservable:
    def __init__(self,
                 p = 2,
                 q = 0.8,
                 nSV = 2,
                 nU = 1,
                 huge_p_matrix = None):
        self.p = p
        self.q = q
        self.nSV = nSV
        self.nU = nU
        self.p_matrix = huge_p_matrix
        self.pq_matrix = None

    @property
    def p_matrix(self):
        return self._p_matrix

    @p_matrix.setter
    def p_matrix(self, value):
        if value is not None:
            self._p_matrix = value
        elif (self.p**(self.nSV) - 1) > 9.0e+18:
            raise ValueError("number of state variables and p value combination exceeds maximum size")
        else:
            # preallocate the final matrix
            pm = np.zeros((self.nSV, (self.p + 1)**(self.nSV)), dtype=int)
            for col in range((self.p + 1)**(self.nSV)):
                base_string = np.base_repr(col, self.p + 1)
                base_string = '0' * (self.nSV - len(base_string)) + base_string
                pm[:,col] = np.flip([int(x) for x in base_string])
        self._p_matrix = pm
        # Add the effect of U

    @property
    def pq_matrix(self):
        return self._pq_matrix
    @pq_matrix.setter
    def pq_matrix(self, value):
        if value is not None:
            self._pq_matrix = value
        else:
            # trim according to q
            pqm = self.p_matrix[:,np.linalg.norm(self.p_matrix, ord=self.q, axis=0) <= self.p]
            # add the rows and columns of the input
            self._pq_matrix = np.concatenate((np.concatenate((pqm, np.zeros([self.nSV, self.nU])), axis=1),np.concatenate((np.zeros([self.nU, pqm.shape[1]]), np.eye(self.nU)), axis=1)), axis=0)


                #     @property
    def pq_function(self):
        pass # To be implemented by the different polynomial classes  

    def __eq__(self, other):
        equal = False
        # If the shapes are not equal, they are definitely not equal
        if self.pq_matrix.shape == other.pq_matrix.shape:
            # If the shpes are equal then... compare the whole array
            equal = (self.pq_matrix == other.pq_matrix).all()
            
        return equal


class hermiteObs(pqObservable):
    @property
    def pq_function(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store the polynomials
        polys = zeros(*indexes.shape)
        for idx, sv in enumerate(x[:-self.nU]):
            polys[idx,:] = Matrix([op.hermite_poly(power,sv) for power in indexes[idx,:]]).transpose()
        for idx, u in enumerate(x[-self.nU:]):
            # Legendre polys in the first order are the constant function l^1(x)=x, perfect for the inputs
            polys[idx + self.nSV,:] = Matrix([op.legendre_poly(power,u) for power in indexes[idx + self.nSV,:]]).transpose()
        # take the product per column
        poly_prod = Matrix([math.prod(polys[:,col]) for col in range(polys.shape[1])]).transpose()
        # lambdify...
        function = lambdify([x], poly_prod, modules='numpy')
        return function

class laguerreObs(pqObservable):
    @property
    def pq_function(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store the polynomials
        polys = zeros(*indexes.shape)
        for idx, sv in enumerate(x[:-self.nU]):
            polys[idx,:] = Matrix([op.laguerre_poly(power,sv) for power in indexes[idx,:]]).transpose()
        for idx, u in enumerate(x[-self.nU:]):
            # Legendre polys in the first order are the constant function l^1(x)=x, perfect for the inputs
            polys[idx + self.nSV,:] = Matrix([op.legendre_poly(power,u) for power in indexes[idx + self.nSV,:]]).transpose()
        # take the product per column
        poly_prod = Matrix([math.prod(polys[:,col]) for col in range(polys.shape[1])]).transpose()
        # lambdify...
        function = lambdify([x], poly_prod, modules='numpy')
        return function
        
class chebyshevtObs(pqObservable):
    @property
    def pq_function(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store the polynomials
        polys = zeros(*indexes.shape)
        for idx, sv in enumerate(x[:-self.nU]):
            polys[idx,:] = Matrix([op.chebyshevt_poly(power,sv) for power in indexes[idx,:]]).transpose()
        for idx, u in enumerate(x[-self.nU:]):
            # Legendre polys in the first order are the constant function l^1(x)=x, perfect for the inputs
            polys[idx + self.nSV,:] = Matrix([op.legendre_poly(power,u) for power in indexes[idx + self.nSV,:]]).transpose()
        # take the product per column
        poly_prod = Matrix([math.prod(polys[:,col]) for col in range(polys.shape[1])]).transpose()
        # lambdify...
        function = lambdify([x], poly_prod, modules='numpy')
        return function

class chebyshevuObs(pqObservable):
    @property
    def pq_function(self):
        x = symbols(f'x:{self.nSV + self.nU}')
        indexes = self.pq_matrix
        # create a numpy array the same size as the indexes to store the polynomials
        polys = zeros(*indexes.shape)
        for idx, sv in enumerate(x[:-self.nU]):
            polys[idx,:] = Matrix([op.chebyshevu_poly(power,sv) for power in indexes[idx,:]]).transpose()
        for idx, u in enumerate(x[-self.nU:]):
            # Legendre polys in the first order are the constant function l^1(x)=x, perfect for the inputs
            polys[idx + self.nSV,:] = Matrix([op.legendre_poly(power,u) for power in indexes[idx + self.nSV,:]]).transpose()
        # take the product per column
        poly_prod = Matrix([math.prod(polys[:,col]) for col in range(polys.shape[1])]).transpose()
        # lambdify...
        function = lambdify([x], poly_prod, modules='numpy')
        return function

if __name__ == "__main__":
    nSV = 3
    nU = 2
    obs = chebyshevtObs(nSV=nSV,p=4,q=0.5,nU=nU)
    obs_o = chebyshevtObs(nSV=nSV,p=4,q=0.5,nU=nU)
    print(obs==obs_o)
    print(obs.pq_function([1,2,3,4,5]))
# sv = 8
# x = symbols(f'x:{sv}')
# hp = op.hermite_poly(1,x[0])
# hpf = lambdify(x,hp)

