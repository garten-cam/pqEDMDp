
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
from pathlib import Path
path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))
from source import pqEDMD
import numpy as np

 
rng = np.random.default_rng(1234)
num_ics = 10
ics_width = 10
ics = ics_width*rng.random((num_ics, 2)) - ics_width/2
 
t_end = 30
n_points = 301
 
# # fit the alg
tr = range(0, 6)
ts = range(6, num_ics)
 
# Test the code for inputs
# use some random input
inputs = 100*rng.random((num_ics, 1)) - 50
# The inputs will be a step that is also a random number
 
def duffode_u(x, t, u):
    # Duffing with two AS points
    return [x[1], -0.5*x[1] + x[0] - x[0]**3 + u]
 
# preallocate the samples list
samples_u = [{'sv': np.empty((n_points, 2)),
              't': np.empty((n_points, 1)), 'u': np.empty((n_points, 1))}
             for _ in range(num_ics)]
 
# populate the samples with noise
meas_std = 0.0
for sample in range(num_ics):
    t = np.linspace(0, t_end, n_points)
    sol = odeint(duffode_u, ics[sample, :], t, args=(inputs[sample][0],))
    samples_u[sample]['sv'] = sol + \
        np.random.normal(0, meas_std, (n_points, 2))
    samples_u[sample]['t'] = t
    samples_u[sample]['u'] = np.full((n_points, 1), inputs[sample][0])
 
duff_EDMD = pqEDMD(p=[5, 7], q=[0.5, 1],
                   polynomial='Legendre',
                   method="rrr")

duff_decomps_u = duff_EDMD.fit([samples_u[i] for i in tr])

err = np.zeros((len(duff_decomps_u),1))
for decomp in range(len(duff_decomps_u)):
    err[decomp] = duff_decomps_u[decomp].error([samples_u[i] for i in ts])

best_duff_decp = np.argmin(err)
 
app_u = duff_decomps_u[best_duff_decp].pred_from_test([samples_u[i] for i in ts])
 
[plt.plot(samples_u[i]['sv'][:, 0], samples_u[i]['sv'][:, 1], 'r')
 for i in tr]
[plt.plot(samples_u[i]['sv'][:, 0], samples_u[i]['sv'][:, 1], 'b')
 for i in ts]
[plt.plot(app_u[i]['sv'][:, 0], app_u[i]
          ['sv'][:, 1], 'k-.') for i in range(len(ts))]
plt.show()
x = 1