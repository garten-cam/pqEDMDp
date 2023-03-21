'''
Author: Camilo Garcia-Tenorio
Code for cleaning the samples that come from rough db.
'''
import db_from_inputs as idb
import numpy as np
import pandas as pd

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

    # Small test here
    def perShaftData(self, rough_samples=idb.dbImport().akima_kiln_samples(), shaft=1):
        # Returns two structures, shaft_01 data and shaft_02 data
        # This first iteration will not consider N shafts, in case htere are three shafts.
        digits = cleanSamples.getDigits(rough_samples)
        # changes the digit to false if it does not meet the selection criteria
        # Match the keys in the rough samples to the keys in the states
        neg_checklist = rough_samples[0].keys()[rough_samples[0].keys().str.contains("|".join(self.states))]
        for idx, sample in enumerate(rough_samples):
            if digits[idx]:
                digits[idx] = cleanSamples.is_proper_sample(sample, shaft=shaft, neg_checklist=neg_checklist)
            
        # preallocate the final 
        
        
        return digits
    # All of them will need to check if the running digit was on
    @staticmethod
    def getDigits(rough_samples):
        # Calls all the filtering methods and performs an and
        # 1. Is the kiln running
        running_digit = cleanSamples.checkRunningDigit(rough_samples)
        # 2. This is tricky... not-not-double burning, returns True if the sample
        # should be included
        inclusive_burning_digit = cleanSamples.checkDoubleBurning(rough_samples)
        # Ok, these are the two things included in matlab. 
        selection_digit = np.logical_and(running_digit, inclusive_burning_digit)
        # If I include more criteria, I have to consider a function that gives the AND
        # of all the available digits
        return selection_digit


    def checkRunningDigit(rough_samples):
        is_running = np.zeros(len(rough_samples))
        for idx, rough_sample in enumerate(rough_samples):
            if rough_sample['BE.AI.SP.KL1_kilnrun_dig'].all():
                is_running[idx] = 1
        is_running.astype(bool)
        return is_running
    
    def checkDoubleBurning(rough_samples):
        # The opposite from matlab to keep consistency of the code.
        # Returns True if the sample should be included
        double_burn = np.zeros(len(rough_samples))
        for idx, rough_sample in enumerate(rough_samples):
            if np.greater(rough_sample['BE.AI.SP.KL1_burning_dig_cc'].diff().ge(1).sum(),1):
                double_burn[idx] = 1
        not_double = np.logical_not(double_burn)
        return not_double
    
    def is_proper_sample(rough_sample, shaft, neg_checklist):
        # Here we can put all the implementation of the selection of samples.
        # Any criterion that can exclude a sample can be coded here.
        # The first I am using is the consitency of combustion shaft, if there is a change in 
        # during the cylce, the sample get excluded.
        proper = dict()
        if rough_sample['Combustion shaft'].all() == shaft:
            proper['shaft'] = True
        else:
            proper['shaft'] = False
        
        for variable in neg_checklist:
            print(rough_sample[variable])
        # Check for negative values
        return proper
    

if __name__ == "__main__":
    clean_samples = cleanSamples()
    print(clean_samples.perShaftData())