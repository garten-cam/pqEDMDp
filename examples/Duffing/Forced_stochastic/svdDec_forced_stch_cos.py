from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pqedmdp import pqEDMDp
import pqobservable as pqo
import decompositions.svddecomposition as svd

rng = np.random.default_rng(73)
num_ics = 6
ics_width = 4
ics = ics_width * rng.random((num_ics, 2)) - ics_width / 2

tfin = 30
n_points = 7*tfin + 1


def duffodeu(x, t, u):
    # Duffing with two AS points
    return [x[1], -0.5 * x[1] + x[0] - x[0] ** 3 + np.cos(u*t)]


# Parameters for the forced simulation. Use step response
u = 4*rng.random((num_ics, 1))

# exp as in experiments. An array of dictionaries. In this case, the number of
# points per experiment is the same, but that is not always the case.
exp = [{
    "y": np.empty((n_points, 2)),
    "y_det": np.empty((n_points, 2)),
    "u": np.empty((n_points, 1)),
    "t": np.empty((n_points, 1))
} for i in range(num_ics)]


for ic, expi, ui in zip(ics, exp, u):
    t = np.linspace(0, tfin, n_points)
    sol = odeint(duffodeu, ic, t, args=(ui[0],))  # How to avoid this indexing?
    expi["y_det"] = sol
    expi["y"] = sol + np.random.normal(0, 0.02, sol.shape)
    expi["u"] = np.cos(ui*t)[..., None]
    expi["t"] = t

# Create the pqEDMD object
pqe = pqEDMDp(
    p=[2, 3, 4],          # Sweep over there 3 values of max order p
    q=[0.5, 1, 1.5, 2],   # Sweep over these q-quasi-norms
    obs=pqo.legendreObs,  # Use legendre observables orthogonal [-inf, inf]
    dyn_dcp=svd.svdDecomposition)

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
fig, axs = plt.subplots(num_ics, sharex=True)
det = [axs[i].plot(expi['t'], expi['y'], 'g', lw=2)
       for i, expi in enumerate(exp)]
# Plot training set
trp = [axs[tri].plot(expi['t'], expi['y'], 'b')
       for expi, tri in zip([exp[i] for i in tr], tr)]
# Plot the testing set
tsp = [axs[tsi].plot(expi['t'], expi['y'], 'r')
       for expi, tsi in zip([exp[i] for i in ts], ts)]
# Plot the approximation
app = [axs[tsi].plot(expi['t'], appxi['y'], '.-k')
       for appxi, expi, tsi in zip(appx, [exp[i] for i in ts], ts)]
axs[-1].set_xlabel('$t$')
[axs[i].set_ylabel('$x$') for i in range(num_ics)]
axs[0].set_title(f"svdDecomposition \n p={dcp.observable.obs_p}, q={
    dcp.observable.obs_q}, n={dcp.sys_n}, $\\epsilon$={np.nanmin(arr):.3f}")
plt.legend((det[0][0], trp[0][0], tsp[0][0], app[0][0]),
           ['Deterministic', 'Training', 'Testing', 'Approx'])
plt.show()
