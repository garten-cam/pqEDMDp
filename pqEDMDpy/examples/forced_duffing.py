# from matplotlib.colors import same_color
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pqedmdpy import pqObservable as pqo
# from pqedmdpy.decompositions.pqdecomposition import pqDecomposition
# from pqedmdpy.decompositions.svddecomposition import svdDecomposition
from pqedmdpy.decompositions.siddecomposition import sidDecomposition
from pqedmdpy.decompositions.sidolsdecomposition import sidOlsDecomposition
from pqedmdpy.pqEDMDp import pqEDMDp

rng = np.random.default_rng(12345)
num_ics = 10
ics_width = 5
ics = ics_width * rng.random((num_ics, 2)) - ics_width / 2

t_end = 30
n_points = 301

# # fit the alg
tr = [1, 2, 3, 4, 6, 8]
ts = [0, 5, 7, 9]

# Test the code for inputs
# use some random input
inputs = 3 * rng.random((num_ics, 1)) - 1
# The inputs will be a step that is also a random number


def duffode_u(x, t, u):
    # Duffing with two AS points
    return [x[1], -0.5 * x[1] + x[0] - x[0] ** 3 + u * np.cos(u * t)]


# preallocate the samples list
samples_u = [
    {
        "y": np.empty((n_points, 2)),
        "t": np.empty((n_points, 1)),
        "u": np.empty((n_points, 1)),
    }
    for _ in range(num_ics)
]

# populate the samples with noise
meas_std = 0.0
for sample in range(num_ics):
    t = np.linspace(0, t_end, n_points)
    sol = odeint(duffode_u, ics[sample, :], t, args=(inputs[sample][0],))
    samples_u[sample]["y"] = sol + np.random.normal(0, meas_std, (n_points, 2))
    samples_u[sample]["t"] = t
    # samples_u[sample]["u"] = np.full((n_points, 1), inputs[sample][0])
    samples_u[sample]["u"] = (
        inputs[sample][0] * np.cos(inputs[sample][0] * samples_u[sample]["t"])
    ).reshape(-1, 1)

EDMD = pqEDMDp(p=[1, 2, 3, 4], q=[0.5, 0.6, 0.7],
               obs=pqo.laguerreObs, dyn_dcp=sidDecomposition)

dcps = EDMD.fit([samples_u[i] for i in tr])

# Calculate the error
# plt.figure(1)
# # for sample in [samples_u[i] for i in tr]:
# #     plt.plot(sample["y"][:, 0], sample["y"][:, 1], 'b')
# for sample in [samples_u[i] for i in ts]:
#     plt.plot(sample["y"][:, 0], sample["y"][:, 1], "r")
#
# plt.show()
# I want to instantiate a partially gerenated object inside the class.
poly = pqo.legendreObs(l=2, p=3, q=1)
# test the decomposition
# Development of the pqEDMD wrapper
decpol = pqo.laguerreObs()
decp = sidDecomposition()

# dec = pqDecomposition(poly, [samples_u[i] for i in tr])
# # # test the svd decomposition
# sdec = svdDecomposition(poly, [samples_u[i] for i in tr])
# # test the sid decomosition
sidd = sidDecomposition(poly, [samples_u[i] for i in tr])
# Test the decomposition
sido = sidOlsDecomposition(poly, [samples_u[i] for i in tr])

#
# err_dec = dec.error([samples_u[i] for i in ts])
# err_sdec = sdec.error([samples_u[i] for i in ts])
err_sidd = sidd.error([samples_u[i] for i in ts])
err_sido = sido.error([samples_u[i] for i in ts])

# pred_dec = dec.predict_from_test([samples_u[i] for i in ts])
# pred_sdec = sdec.predict_from_test([samples_u[i] for i in ts])
pred_sidd = sidd.predict_from_test([samples_u[i] for i in ts])
pred_sido = sido.predict_from_test([samples_u[i] for i in ts])


# plt.figure(1)
# # for sample in [samples_u[i] for i in tr]:
# #     plt.plot(sample["y"][:, 0], sample["y"][:, 1], 'b')
# for sample in [samples_u[i] for i in ts]:
#     plt.plot(sample["y"][:, 0], sample["y"][:, 1], "r")
# for prediction in pred_dec:
#     plt.plot(prediction["y"][:, 0], prediction["y"][:, 1], "k")
#
# plt.figure(2)
# for sample in [samples_u[i] for i in ts]:
#     plt.plot(sample["y"][:, 0], sample["y"][:, 1], "r")
# for prediction in pred_sdec:
#     plt.plot(prediction["y"][:, 0], prediction["y"][:, 1], "k")
#
plt.figure(3)
for sample in [samples_u[i] for i in ts]:
    plt.plot(sample["y"][:, 0], sample["y"][:, 1], "r")
for prediction in pred_sidd:
    plt.plot(prediction["y"][:, 0], prediction["y"][:, 1], "k")

plt.figure(4)
for sample in [samples_u[i] for i in ts]:
    plt.plot(sample["y"][:, 0], sample["y"][:, 1], "r")
for prediction in pred_sido:
    plt.plot(prediction["y"][:, 0], prediction["y"][:, 1], "k")

sidd.spectrum()
plt.show()


duff_decomps_u = duff_EDMD.fit([samples_u[i] for i in tr])

err = np.zeros((len(duff_decomps_u), 1))
for ind, decomp in enumerate(duff_decomps_u):
    err[ind] = decomp.error([samples_u[i] for i in ts])

best_duff_decp = np.argmin(err)

app_u = duff_decomps_u[best_duff_decp].pred_from_test(
    [samples_u[i] for i in ts])

[plt.plot(samples_u[i]["sv"][:, 0], samples_u[i]["sv"][:, 1], "r") for i in tr]
[plt.plot(samples_u[i]["sv"][:, 0], samples_u[i]["sv"][:, 1], "b") for i in ts]
[plt.plot(app_u[i]["sv"][:, 0], app_u[i]["sv"][:, 1], "k-.")
 for i in range(len(ts))]
plt.show()
x = 1
