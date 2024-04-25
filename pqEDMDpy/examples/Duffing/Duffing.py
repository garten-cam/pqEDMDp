from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from pqedmdpy.pqEDMDp import pqEDMDp
from pqedmdpy import pqObservable as pqo
from pqedmdpy.decompositions.siddecomposition import sidDecomposition

rng = np.random.default_rng(12345)
num_ics = 10
ics_width = 5
ics = ics_width * rng.random((num_ics, 2)) - ics_width / 2

t_end = 30
n_points = 301

# # fit the alg
tr = [1, 2, 3, 4, 6, 8]
ts = [0, 5, 7, 9]


def duffode(x, t):
    # Duffing with two AS points
    return [x[1], -0.5 * x[1] + x[0] - x[0] ** 3]


# exp as in experiments. An array of dicitonaries. in this case the number of
# points per experiment is the same, but that is not always the case.
exps = [{
    "y": np.empty((n_points, 2)),
    "t": np.empty((n_points, 1))
} for _ in range(num_ics)]

# numerical integration of the oscillator from different initial conditions
for i, exp in enumerate(exps):
    t = np.linspace(0, t_end, n_points)
    sol = odeint(duffode, ics[i, :], t)
    exp["y"] = sol
    exp["t"] = t

# Create the decomposition wrapper

sidREDMD = pqEDMDp(p=[2, 3, 4], q=[1.5, 2],
                   obs=pqo.laguerreObs, dyn_dcp=sidDecomposition)
dcps = sidREDMD.fit([exps[i] for i in tr])
