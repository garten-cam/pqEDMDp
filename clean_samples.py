'''
Author: Camilo Garcia-Tenorio
Code for cleaning the samples that come from rough db.
'''
import db_from_inputs as idb
import numpy as np
import matplotlib.pyplot as plt

class cleanSamples:
    def __init__(self,
                 states = ['CO-raw-stack_ana',
                           'CO2-raw-stack_ana',
                           'NO_raw-stack_ana',
                           'O2-raw-stack_ana',
                           'T1-fumes_ana',
                           'T1-lime-A_ana',
                           'T1-lime-B_ana',
                           'T2-fumes_ana',
                           'T2-lime-A_ana',
                           'T2-lime-B_ana',
                           'TT-pyro-S1_ana',
                           'TT-pyro-S2_ana',
                           'TT-pyroS1-S2_ana',
                           'PT-channel_ana',
                           'PT-primary-air_ana',
                           'PT-secondary-air_ana'],
                inputs = ['total-rpm-PA_calc',        
                          'total-rpm-SA_calc',
                          'speed-282_ana',
                          'speed-284_ana',
                          'Speed-AC116_ana',
                          'Speed-AC117_ana',
                          'Speed-AC146_ana',
                          'Speed-AC151_ana',
                          'Speed-AC152_ana',
                          'Speed-AC153_ana',
                          'Speed-AC154_ana',
                          'Speed-AC155_ana',
                          'Speed-AC156_ana',
                          'Speed-AC157_ana',
                          'Speed-AC158_ana',
                          'Speed-AC159_ana',
                          'Speed-AC160_ana',
                          'Speed-AC161_ana',
                          'FA0148-intensity_ana',
                          'FA0148-speed_ana']):
        self.states = states
        self.inputs = inputs
        # If the rough samples are not prvided, the input calls the default method.
        # I wanted to make an inference from the states and inputs but I dont know 
        # the correlation with the 02 and 10 seconds variables. Maybe something that
        # does not belong gets imported. 

    # The three cases I have are: per shaft, one long term, and several long term
    def continousCylces(self, rough_samples=idb.dbImport().akima_kiln_samples()):

        continous_samples = 1
        return continous_samples
    # Selecting for each shaft is done!
    def perShaftData(self, rough_samples=idb.dbImport().akima_kiln_samples(), shaft=1):
        # 0. Because of the changes in the imports, now I need the index the cycles staritgn from 0 and including the 
        # lenght of the rough_samples + 1
        cycle_index = np.concatenate((np.array([0]), np.nonzero(rough_samples['Cycle - Number'].diff().to_numpy(na_value=0))[0], np.array([len(rough_samples)+1])))
        # Returns list of structures, where each structure returns the time vector, 
        # the states matrix and the input matrix 
        # 1. Get the basic selection digits, selects when the kiln is running and there 
        # are no double starts in the burning process
        digits = cleanSamples.getDigits(rough_samples,cycle_index)
        # 2. Get more digits, the ones that signal that is the right shaft.
        # I s this necessary? just to make the "and" operation?
        is_shaft = np.zeros(len(rough_samples))
        for idx in range(cycle_index.size - 1):
            if rough_samples['Combustion shaft'].loc[cycle_index[idx]:cycle_index[idx+1]].all() == shaft:
                is_shaft[cycle_index[idx]:cycle_index[idx+1]] = 1
        
        select_sample = np.logical_and(digits, is_shaft)
        # 3. Prune the sataset
        rough_samples = rough_samples.loc[select_sample]
        # 3. Now, main division loop
        # 3.1. recalculate the cycle index
        cycle_index_shaft = np.concatenate((np.array([0]), np.nonzero(rough_samples['Cycle - Number'].diff().to_numpy(na_value=0))[0], np.array([len(rough_samples)+1])))
        # 3.2. Match the state and input names with the keys in the df
        state_match = rough_samples.keys()[rough_samples.keys().str.contains("|".join(self.states))]
        input_match = rough_samples.keys()[rough_samples.keys().str.contains("|".join(self.inputs))]
        # 3.3. Now....... loop! take only the part of the cycle when the burning digit is guan
        # 3.3.1. match the burning digit
        burn = rough_samples.keys()[rough_samples.keys().str.contains('burning_dig')]
        # 3.3.2. preallocate the samples list
        samples = [dict()]*(cycle_index_shaft.size - 1)
        # 3.3.3. loop
        for sample in range(len(samples)):
            # Bring all the data from that cycle and index further according to the 
            burn_index = rough_samples[burn].iloc[cycle_index_shaft[sample]:cycle_index_shaft[sample+1]].to_numpy().ravel().astype(bool)
            samples[sample]['sv'] = (rough_samples[state_match].iloc[cycle_index_shaft[sample]:cycle_index_shaft[sample+1]].loc[burn_index]).to_numpy()
            samples[sample]['u'] = (rough_samples[input_match].iloc[cycle_index_shaft[sample]:cycle_index_shaft[sample+1]].loc[burn_index]).to_numpy()
            samples[sample]['t'] = rough_samples['timestamp'].iloc[cycle_index_shaft[sample]:cycle_index_shaft[sample+1]].loc[burn_index]
        return samples


    # All of them will need to check if the running digit was on
    @staticmethod
    def getDigits(rough_samples,cycle_index):
        # Calls all the filtering methods and performs an and
        # 1. Is the kiln running
        digits = dict()
        digits['running_digit'] = cleanSamples.checkRunningDigit(rough_samples, cycle_index)
        # 2. This is tricky... not-not-double burning, returns True if the sample
        # should be included
        digits['inclusive_burning_digit'] = cleanSamples.checkDoubleBurning(rough_samples, cycle_index)
        # Ok, these are the two things included in matlab. 
        selection_digit = np.logical_and(digits['running_digit'], digits['inclusive_burning_digit'])
        # If I include more criteria, I have to consider a function that gives the AND
        # of all the available digits
        return selection_digit


    def checkRunningDigit(rough_samples, cycle_index):
        is_running = np.zeros(len(rough_samples))
        for idx in range(cycle_index.size - 1):
            if rough_samples['BE.AI.SP.KL1_kilnrun_dig'].loc[cycle_index[idx]:cycle_index[idx+1]].all():
                is_running[cycle_index[idx]:cycle_index[idx+1]] = 1
        is_running.astype(bool)
        return is_running
    
    def checkDoubleBurning(rough_samples, cycle_index):
        # The opposite from matlab to keep consistency of the code.
        # Returns True if the sample should be included
        double_burn = np.zeros(len(rough_samples))
        for idx in range(cycle_index.size - 1):
            if np.greater(rough_samples['BE.AI.SP.KL1_burning_dig_cc'].loc[cycle_index[idx]:cycle_index[idx+1]].diff().ge(1).sum(),1):
                double_burn[cycle_index[idx]:cycle_index[idx+1]] = 1
        not_double = np.logical_not(double_burn)
        return not_double
    
    # def is_proper_sample(rough_sample, shaft):
    #     # Here we can put all the implementation of the selection of samples.
    #     # Any criterion that can exclude a sample can be coded here.
    #     # The first I am using is the consitency of combustion shaft, if there is a change in 
    #     # during the cylce, the sample get excluded.
    #     proper = dict()
    #     if rough_sample['Combustion shaft'].all() == shaft:
    #         proper['shaft'] = True
    #     else:
    #         proper['shaft'] = False
    #     # This is the space to add more criteria, this is for per sample cleaning
    #     # Check for negative values
    #     return proper['shaft']
    

if __name__ == "__main__":
    clean_samples = cleanSamples()
    print(clean_samples.perShaftData())