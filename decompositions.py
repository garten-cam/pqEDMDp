'''
Author: Camilo Garcia Tenorio
returns a decomposition object
'''
class decompositions:
    def __init__(
            self,
            observable,
            ):
        # Generic class for defining a decomposition
        self.observable = observable
        self.matrixA = None
        self.B = None
        self.C = None
        # self.inverse = None

    def calc_U(self, xtr, ytr):
        # save the observable function from the object provided at the init
        # method
        psi = self.observable.pq_function
        # Evaluate the training data in x and y
        x_eval = [psi(x_row) for x_row in range(xtr)]
        return psi
    @property
    def A(self):
        return self.A

    @property
    def B(self):
        return self.B
    
    @property
    def C(self):
        return self.C



class maxLikeDecomp(decompositions):
    @property
    def A(self):
        return self.A
    
