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
from dataclasses import dataclass


# Import the file
url = 'https://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
data = pd.read_csv(url)

# The file contains 600 datapoits, lets see it
data.plot(x="Time", y=["H1", "H2", "T1", "T2"])
plt.xlabel("Time [s]")
plt.ylabel("Heat Input/Temperature")
plt.title('TcLab samples')
#
# The alg is not working for systems where the samples have different length.
# Draw form an uniform distribution
s_i = np.linspace(0, 600, 7, dtype=int)
diff = [-8, -4, 1, 10, 10]  # Some consistent offset
s_i += np.hstack(([0], diff, [0]))
# Populate the samples, this can and should be a dataclass, making it
# dot-indexable and similar to the Matlab implementation
samples = [{
    't': data['Time'].values[s_i[i - 1]:s_i[i]],
    'y': data[['T1', 'T2']].values[s_i[i - 1]:s_i[i], :],
    'u': data[['H1', 'H2']].values[s_i[i - 1]:s_i[i], :],
} for i in range(1, 7)]

# define the training and testing sets
ts = [2, 4]
tr = [0, 1, 3, 5]

# Plot the training, testing and
# plt.figure(3)
# for smp in [samples[i] for i in tr]:


# define the decomposition wrapers
sidREDMD = pqEDMDp(p=[2, 3], q=[0.6, 1.2],
                   obs=pqo.pqObservable, dyn_dcp=sidDecomposition)
svdEDMD = pqEDMDp(p=[2, 3], q=[0.6, 1.2, 2],
                  obs=pqo.legendreObs, dyn_dcp=svdDecomposition)

# fit them
sidRT = sidREDMD.fit([samples[i] for i in tr])
svdT = svdEDMD.fit([samples[i] for i in tr])

# get the error
err_R = [ri.error([samples[i] for i in ts]) for ri in sidRT]
err_svd = [si.error([samples[i] for i in ts]) for si in svdT]

# test the test
testR = sidRT[np.argmin(err_R)].predict_from_test([samples[i] for i in ts])
testS = svdT[np.argmin(err_svd)].predict_from_test([samples[i] for i in ts])
# add the time for plotting
for itR, tst in zip(testR, [samples[i] for i in ts]):
    itR['t'] = tst['t']
# for plotting purposes, add the time to the output dictionaty
for itS, tst in zip(testS, [samples[i] for i in ts]):
    itS['t'] = tst['t']

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

# plt.figure(2)
# for smp in [samples[i] for i in tr]:
#     plt.plot(smp['t'], smp['y'], 'b')
# for smp in [samples[i] for i in ts]:
#     plt.plot(smp['t'], smp['y'], 'r')
# for smp in testO:
#     plt.plot(smp['t'], smp['y'], 'k')
# plt.xlabel('Time [s]')
# plt.ylabel('Temperature')
# plt.title('sidO Decomposition')

plt.figure(3)
for smp in [samples[i] for i in tr]:
    plt.plot(smp['t'], smp['y'], 'b')
for smp in [samples[i] for i in ts]:
    plt.plot(smp['t'], smp['y'], 'r')
for smp in testS:
    plt.plot(smp['t'], smp['y'], 'k')
plt.xlabel('Time [s]')
plt.ylabel('Temperature')
plt.title('svd Decomposition')
plt.savefig(
    '/home/cgarcia/Documents/FinalReportBW/FinalReportBW/figures/TCLabAprox.eps', format='eps')
plt.show()
