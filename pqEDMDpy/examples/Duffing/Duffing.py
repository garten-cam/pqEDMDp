from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pqedmdpy.pqEDMDp import pqEDMDp
from pqedmdpy import pqObservable as pqo
from pqedmdpy.decompositions.siddecomposition import sidDecomposition

rng = np.random.default_rng(73731167)
num_ics = 9
# ics_width = 2
# ics = ics_width * rng.random((num_ics, 2)) - ics_width / 2
ics = np.array(
    [[-0.2500, 1.5149],
     [-1.9945, - 0.5408],
        [-1.1674, 0.6747],
        [-0.9247, - 0.6895],
        [-1.7546, 1.1917],
        [-0.5441, - 0.7326],
        [1.2276, 0.9499],
        [-1.2993, - 0.2923],
        [-0.0619, 1.8947]])

t_end = 30
n_points = 301

# # fit the alg
tr = [0, 1, 3, 5, 7, 8]
ts = [2, 4, 6]


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

# Create the decomposition wrapper: with some observable, and some decomposition

sidEDMD = pqEDMDp(p=[2, 3, 4], q=[0.5, 1, 1.5, 2],
                  obs=pqo.legendreObs, dyn_dcp=sidDecomposition)
# Fit it, i.e., get the decompositions
dcps = sidEDMD.fit([exps[i] for i in tr])

# Calculate the error
err = [dp.error([exps[i] for i in ts]) for dp in dcps]

# The best one
bst = np.argmin(err)

test = dcps[bst].predict_from_test([exps[i] for i in ts])

plt.figure(1)
for smp in [exps[i] for i in tr]:
    plt.plot(smp['y'][:, 0], smp['y'][:, 1], 'b')
for smp in [exps[i] for i in ts]:
    plt.plot(smp['y'][:, 0], smp['y'][:, 1], 'r')
for smp in test:
    plt.plot(smp['y'][:, 0], smp['y'][:, 1], '.k')

plt.show()
