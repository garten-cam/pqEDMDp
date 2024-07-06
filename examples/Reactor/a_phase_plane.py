'''
Implementation of the pqEDMD for the small reactor example
'''
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pqedmdpy.pqEDMDp import pqEDMDp
from pqedmdpy import pqObservable as pqo

# Define the ODE


def cstr_ode(x, y, d) -> np.ndarray[float]:
    """
    cstr_ode returns the vector field of the simple reactor example 
    """
