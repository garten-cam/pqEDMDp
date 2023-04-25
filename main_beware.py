'''
Author: Camilo Garcia-Tenorio
Main script for the Beware development.
'''
# import db_from_inputs as idb
# import clean_samples as cls
import pqEDMD as pqe
import pickle

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
with open('samples.plk', 'rb') as file_handle:
    samples = pickle.load(file_handle)

# Next, continue developing. call the pqEDMD
# cretate the rough object
pqEDMD_o = pqe.pqEDMD(p=[1], q=[0.4, 0.6], polynomial='Laguerre')

# Ok, call the function fit in the script
pq_approximations = pqEDMD_o.fit(samples[:-10])
