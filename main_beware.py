'''
Author: Camilo Garcia-Tenorio
Main script for the Beware development. 
'''
import db_from_inputs as idb
import clean_samples as cls

import_obj = idb.dbImport() # create the object with the info
rough_samples = import_obj.akima_kiln_samples()

# clean the samples according to the prefered method
samples = cls.cleanSamples()
samples = samples.perShaftData(rough_samples)
print(rough_samples)