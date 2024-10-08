from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pqedmdp import pqEDMDp
import pqobservable as pqo
import decompositions.siddecomposition as sid
rng = np.random.default_rng(73)
num_ics = 6
ics_width = 4
ics = ics_width * rng.random((num_ics, 2)) - ics_width / 2

tfin = 30
n_points = 5*tfin + 1


def duffode(x, t):
    # Duffing with two AS points
    return [x[1], -0.5 * x[1] + x[0] - x[0] ** 3]


# exp as in experiments. An array of dictionaries. In this case, the number of
# points per experiment is the same, but that is not always the case.
exp = [{
    "y": np.empty((n_points, 2)),
    "y_det": np.empty((n_points, 2)),
    "t": np.empty((n_points, 1))
} for _ in range(num_ics)]


for ic, expi in zip(ics, exp):
    t = np.linspace(0, tfin, n_points)
    sol = odeint(duffode, ic, t)
    expi["y_det"] = sol
    expi["y"] = sol + np.random.normal(0, 0.02, sol.shape)
    expi["t"] = t

# Create the pqEDMD object
pqe = pqEDMDp(
    p=[2, 3, 4],          # Sweep over there 3 values of max order p
    q=[0.5, 1, 1.5, 2],   # Sweep over these q-quasi-norms
    obs=pqo.legendreObs,  # Use legendre observables orthogonal [-inf, inf]
    dyn_dcp=lambda obs, sys: sid.sidDecomposition(4, 3, obs, sys))

# Define the indexes for the training and testing sets
tr = [1, 2]
ts = [0, 3, 4, 5]

# Fit the model with the training set
dcps = pqe.fit([exp[i] for i in tr])

# Calculate the error with the test set
err = [dcp.error([exp[i] for i in ts]) for dcp in dcps]
# For reference, we can have the absolute error
arr = [dcp.abs_error([exp[i] for i in ts]) for dcp in dcps]

# Get the best decomposition
dcp = dcps[np.nanargmin(arr)]

# Predict from the best decomposition
appx = dcp.predict_from_test([exp[i] for i in ts])

plt.ioff()
fig = plt.figure(2)
# Plot the deterministic states
det = [plt.plot(expi['y_det'][:, 0], expi['y'][:, 1], 'g', lw=2)
       for expi in exp]
# Plot training set
trp = [plt.plot(expi['y'][:, 0], expi['y'][:, 1], 'b')
       for expi in [exp[i] for i in tr]]
# Plot the testing set
tsp = [plt.plot(expi['y'][:, 0], expi['y'][:, 1], 'r')
       for expi in [exp[i] for i in ts]]
# Plot the approximation
app = [plt.plot(appxi['y'][:, 0], appxi['y'][:, 1], '.-k')
       for appxi in appx]
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend((det[0][0], trp[0][0], tsp[0][0], app[0][0]),
           ['Deterministic', 'Training', 'Testing', 'Approx'])
plt.title(f"sidDecomposition \n p={dcp.observable.obs_p}, q={
          dcp.observable.obs_q}, $\\epsilon$={np.nanmin(arr):.3f}")
plt.show()
