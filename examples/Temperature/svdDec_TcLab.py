import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pqedmdp import pqEDMDp
import pqobservable as pqo
import decompositions.svddecomposition as svd

# Read the data
data = pd.read_csv(f"{os.path.dirname(__file__)}/dyn_data.csv")

plt.ioff()
# plt.figure(1)
data.plot(x="Time", y=["H1", "H2", "T1", "T2"])
plt.xlabel("Time [s]")
plt.ylabel("Heat Input/Temperature")
plt.title('TcLab samples')

s_i = np.linspace(0, 600, 7, dtype=int)
diff = [-8, -4, 1, 10, 10]  # Some consistent offset
s_i += np.hstack(([0], diff, [0]))
# Extract the experiments. One set per heater
# This is assuming that each of the heaters has an input related to the voltage
# and an input related to the temperature of the neighbor
exp = [{
    't': data['Time'].values[s_i[i - 1]:s_i[i]],
    'y': data[['T1', 'T2']].values[s_i[i - 1]:s_i[i], :],
    'u': data[['H1', 'H2']].values[s_i[i - 1]:s_i[i], :],
} for i in range(1, 7)]
# Define the training and testing sets
ts = [2, 4]
tr = [0, 1, 3, 5]
# Define the pq
pqe = pqEDMDp(
    p=[2, 3, 4],
    q=[0.6, 1.2, 1.5],
    obs=pqo.legendreObs,
    dyn_dcp=svd.svdDecomposition)

# Fit the decompositions
dcps = pqe.fit([exp[i] for i in tr])

# Calculate the error with the test set
err = [dcp.error([exp[i] for i in ts]) for dcp in dcps]

# Get the best decomposition
dcp = dcps[np.nanargmin(err)]

# Predict from the best decomposition
appx = dcp.predict_from_test([exp[i] for i in ts])

# Plots
plt.figure(2)
# Traning set
trp = [plt.plot(expi['t'], expi['y'], 'b') for expi in [exp[i] for i in tr]]
# Testing set
tsp = [plt.plot(expi['t'], expi['y'], 'r') for expi in [exp[i] for i in ts]]
# Approximation
app = [plt.plot(expi['t'], appi['y'], '-.k')
       for expi, appi in zip([exp[i] for i in ts], appx)]
# inputs
inpH1 = [plt.plot(expi['t'], expi['u'][:, 0], 'c') for expi in exp]
inpH1 = [plt.plot(expi['t'], expi['u'][:, 1], 'm') for expi in exp]
plt.show()
