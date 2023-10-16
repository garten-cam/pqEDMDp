# %%
'''
Author: Camilo Garcia-Tenorio
Main script for the Beware development.
'''
import os
import sys
import matplotlib.pyplot as plt
from source import pqEDMD as pqd

# We need to bring in the data.
# Therefore, we are using the AK_forecasting and AK_CTA 
# Now, finally. Import the data
# it all starts with the data loader...
# I am not going to generate a pipeline, the import conflicts are not letting me get so far.
# So... I am leaving here code that must be run to get the set of cycles into a database on the local drive...  
cba_source = 'bemaap121'
path = r'C:\Users\cgarcia\simulations'
folder = 'test_02102023'

dll = dl.DataLoader(kiln='BEAIKL1',
                    nb_sequence_cycles=30, # batches of 30 cons cycles
                    nb_required_seq=10, # How many sequences per cluster
                    min_cycle_wo_dt=20, # Number of cycles w/o downtime
                    max_dt_val=0, # 
                    cba_source=cba_source,
                    folder_path=path, folder_name=folder)
                    
selection = dll.run_loader(verbose=False)
# This returns a json file to the specified directory. The jason file that
# contains the cluster, and the batches of cycles that fit the rules 
# %%
# Now, we move to the data extractor
kiln = 'BEAIKL1'
branch = None
source_dir = 'prod'
path = r'C:\Users\cgarcia\simulations'
folder = 'test_02102023'
sets_per_cluster = 2 
sensors_dict = {'2s':
                        ['BE.AI.SP.KL1_loads-counter_calc',
                         'BE.AI.SP.KL1_NGBOT_dig',
                         'BE.AI.SP.KL1_PzTime_dig',
                         'BE.AI.SP.KL1_SFBOT_dig',
                         'BE.AI.SP.KL1_burning_dig_cc',
                         'BE.AI.SP.KL1_kilnrun_dig', 'BE.AI.SP.KL1_reversal_dig',
                         'BE.AI.SP.KL1_stack-anal-clean_dig',
                         'BE.AI.SP.KL1_stand-by_dig'],
                    '10s': [
                        "BE.AI.SP.KL1_Speed-AC116_ana",
                        "BE.AI.SP.KL1_Speed-AC117_ana",
                        "BE.AI.SP.KL1_Speed-AC146_ana",
                        "BE.AI.SP.KL1_PT-primary-air_ana",
                        "BE.AI.SP.KL1_PT-channel_ana",
                        "BE.AI.SP.KL1_PT-secondary-air_ana"
                    ]
                    }
de = dex.DataExtractor(kiln, path, folder, sets_per_cluster, sensors=sensors_dict)
de.run_extractor()

# %%
