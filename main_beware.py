'''
Author: Camilo Garcia-Tenorio
Main script for the Beware development.
'''
import db_from_inputs as idb
import clean_samples as cls
import pqEDMD as pqe
import pickle
import matplotlib.pyplot as plt

# import_obj = idb.dbImport() # create the object with the info
# rough_samples = import_obj.akima_kiln_samples()

# # clean the samples according to the prefered method
# samples = cls.cleanSamples()
# samples = samples.perShaftData(rough_samples)
# with open('samples.pkl', 'wb') as file_handle:
#     pickle.dump(samples, file_handle)
# print(samples)

# I can import the data easitly from the data import classes
# Now I need a quick method to retrieve them. In the previous lines, I
# saved the pickle... Now, retrieve it
with open('samples.pkl', 'rb') as file_handle:
    samples = pickle.load(file_handle)

# Next, continue developing. call the pqEDMD
# cretate the rough object
pqEDMD_o = pqe.pqEDMD(p=[2], q=[1],
                      polynomial='Hermite',
                      method="",
                      normalization=True)

# # Ok, call the function fit in the script
pq_approximations = pqEDMD_o.fit(samples[20:40])

# pq_pred = pqEDMD_o.predict(pq_approximations[0],
#                            [samples[i]['sv'][0, :] for i in range(len(samples)-10,len(samples),1)],
#                            pqEDMD_o.scalers,
#                            [len(samples[i]['u']) for i in range(len(samples)-10,len(samples),1)],
#                            [samples[i]['u'] for i in range(len(samples)-10,len(samples),1)])

pq_pred = pqEDMD_o.predict(pq_approximations[0],
                           [samples[i]['sv'][0, :] for i in range(40, 41)],
                           pqEDMD_o.scalers,
                           [len(samples[i]['u']) for i in range(40, 41, 1)],
                           [samples[i]['u'] for i in range(40, 41, 1)])
for vtp in range(pq_approximations[0].observable.nSV):
    plt.figure()
    plt.plot(pq_pred[0]['sv'][:, vtp])
    plt.plot(samples[40]['sv'][:, vtp])
    plt.show()
x = 1
