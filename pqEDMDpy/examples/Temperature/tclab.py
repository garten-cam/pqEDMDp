"""
Script to test the algorithm on real data.
Author: Camilo Garcia Tenorio
"""

from pqedmdpy import pqObservable as pqo
# from pqedmdpy.decompositions.pqdecomposition import pqDecomposition
from pqedmdpy.decompositions.svddecomposition import svdDecomposition
from pqedmdpy.decompositions.siddecomposition import sidDecomposition
from pqedmdpy.decompositions.sidolsdecomposition import sidOlsDecomposition
from pqedmdpy.pqEDMDp import pqEDMDp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import the file
url = 'https://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)

# The file contains 600 datapoits lets see it
# data.plot(x="Time", y=["H1", "H2", "T1", "T2"])
# plt.xlabel("Time [s]")
# plt.ylabel("Heat Input/Temperature")
# plt.title('TcLab samples')
#
# From the 600 datapoints, divide the set into 6 samples of 100 seconds each
s_i = np.linspace(0, 600, 7, dtype=int)
samples = [{
    't': data['Time'].values[s_i[i - 1]:s_i[i]],
    'y': data[['T1', 'T2']].values[s_i[i - 1]:s_i[i], :],
    'u': data[['H1', 'H2']].values[s_i[i - 1]:s_i[i], :],
} for i in range(1, 7)]

# define the training and testing sets
ts = [2]
tr = [0, 1, 3, 4, 5]

# define the decomposition wrapers
sidREDMD = pqEDMDp(p=[3], q=[1],
                   obs=pqo.legendreObs, dyn_dcp=sidDecomposition)
sidOEDMD = pqEDMDp(p=[3], q=[1],
                   obs=pqo.legendreObs, dyn_dcp=sidOlsDecomposition)

# fit them
sidRT = sidREDMD.fit([samples[i] for i in tr])
sidOT = sidOEDMD.fit([samples[i] for i in tr])

# get the error
err_R = [sidRT[ri].error([samples[i] for i in ts]) for ri in range(len(sidRT))]
err_O = [sidOT[oi].error([samples[i] for i in ts]) for oi in range(len(sidOT))]

# test the test
testR = sidRT[np.argmin(err_R)].predict_from_test([samples[i] for i in ts])
testO = sidOT[np.argmin(err_O)].predict_from_test([samples[i] for i in ts])
# add the time for plotting
for itR, tst in zip(testR, [samples[i] for i in ts]):
    itR['t'] = tst['t']
for itO, tst in zip(testO, [samples[i] for i in ts]):
    itO['t'] = tst['t']

# plot the test
plt.figure(1)
for smp in [samples[i] for i in tr]:
    plt.plot(smp['t'], smp['y'], 'b')
for smp in [samples[i] for i in ts]:
    plt.plot(smp['t'], smp['y'], 'r')
for smp in testR:
    plt.plot(smp['t'], smp['y'], 'k')
plt.xlabel('Time [s]')
plt.ylabel('Temperature')
plt.title('sidR Decomposition')

plt.figure(2)
for smp in [samples[i] for i in tr]:
    plt.plot(smp['t'], smp['y'], 'b')
for smp in [samples[i] for i in ts]:
    plt.plot(smp['t'], smp['y'], 'r')
for smp in testO:
    plt.plot(smp['t'], smp['y'], 'k')
plt.xlabel('Time [s]')
plt.ylabel('Temperature')
plt.title('sidO Decomposition')
plt.show()
u = 1
