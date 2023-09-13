'''
Author: Camilo Garcia-Tenorio
Code for returning a database (pandas or sqlite) to perform the data cleaning from the output of this function.
'''
# import sys # maybe use it later for the parser
# import argparse # when it is time for refining the code for deployment we can use this
import sqlite3
import pandas as pd
import numpy as np
"""
Ok, I was starting to develop a code that was able to handle the inputs with a parser such that
the code is suitable for easy deployment in the future via just a terminal. 

While that is a plus for the future, right now, the priority is to get the models working
so all the tasks that I want to do via a parser or directly on a database can wait.

    TODO [X] Import the data into a pandas database from the local sqlite databases and/or the csv.
    TODO [X] Create a list of tables (dfs), where each entry is a distinct cycle. 
"""


class dbImport:
    # Class constructor
    def __init__(self, path_02seconds="C:\Expert System Codes\Prism_extracts\RawData\BEAIKL1_RD_02s_20220501_20220901.db",
                 path_10seconds="C:\Expert System Codes\Prism_extracts\RawData\BEAIKL1_RD_10s_20220501_20220901.db",
                 path_60seconds="C:\Expert System Codes\Prism_extracts\RawData\BEAIKL1_RD_60s_20220101_20220901.db",
                 path_transfos="C:\Expert System Codes\Prism_extracts\Pickle\BEAIKL1_PRISM_DT_Tags_full_20230227.pickle",
                 # path_transfos="C:\Expert System Codes\Prism_extracts\Data\BEAIKL1_PRISM_DT_Tags_full_20221022.csv",
                 start_date="2022-07-07 15:29:00",
                 end_date="2022-07-08 23:27:04",
                 interpolation_method='akima',
                 features_02seconds=['Time',
                                     'Speed',
                                     'WE1314-measured',
                                     'raw',
                                     'PT',
                                     'dig',
                                     'loads',
                                     'strokes',
                                     'timestamp'],
                 features_10seconds=['timestamp',
                                     'level-rad',
                                     'rpm-PA',
                                     'speed',
                                     'Speed',
                                     'TT',
                                     'fumes',
                                     'lime']):
        self.path_02seconds = path_02seconds
        self.path_10seconds = path_10seconds
        self.path_60seconds = path_60seconds
        self.path_transfos = path_transfos
        self.start_date = start_date
        self.end_date = end_date
        self.interpolation_method = interpolation_method
        self.features_02seconds = features_02seconds
        self.features_10seconds = features_10seconds
        # all of the above are self explanatory, except for the last one, the last one is the interpolation method. I will not use the fillna because it does not have fancy polynomials and splines and akima...

    # Copying the same development that I made for matlab, the steps are:
    # TODO [X] then, A function that calls all the inividual importers and concatenates in a single dataframe. or numpy array? In the Matlab code is still a table.
    # At the end, I have just one importer for databases, and one for the pickle
    # TODO [X] Again with the interpolation canundrum.... Makima was working well in matlab.
    # Use Makima and see if the signal can be improved later. Done!!!
    # TODO [ ] Other interpolation methods. Maybe some filtering also
    # TODO [ ] This lacks a functinality that I had in matlab. In matlab I am able to
    # save the file of a particular step in the process and start from there.

    def akima_kiln_samples(self):
        # This method uses akima interpolation and no filtering to divide the data
        kiln_dataframe = dbImport.concatenateDBs(self)  # get the samples
        kiln_dataframe = kiln_dataframe.interpolate(method='akima')
        # get the diff of the leads counter
        # I need to calculate the begginging and the end of a cycle
        # the diff function returns the values one index ahead than matlab.
        # It preserves the shape but the first value is nan meaning that this is
        # the beginning of a cycle.
        # Get the index in which those columns are nonzero
        cycle_index = np.nonzero(
            kiln_dataframe['Cycle - Number'].diff().to_numpy(na_value=0))[0]
        # Python indexing excludes the final element because of the zero indexing...
        # So it is not necessary to calculate an end index
        # I will not perform the division here
        kiln_dataframe = kiln_dataframe.iloc[cycle_index[0]:cycle_index[-1], :]
        return kiln_dataframe

    def concatenateDBs(self):
        # Separate method from the import to keep things uncluttered.
        # The objective is to concatenate in a single database, along
        #
        # 1. import the databases
        kiln_dfs = dbImport.importDBs(self)
        # 2. Create the return database with the innerjoin of the 02
        # seconds file and the transfos
        kiln_dataframe = pd.merge(kiln_dfs['data_02s'], kiln_dfs['transfos'],
                                  left_on='BE.AI.SP.KL1_loads-counter_calc',
                                  right_on='Cycle - Number',
                                  how='inner')
        # The line was not wrong... It was an error in the pickle :/
        # Continue, merge the 10 seconds data... Hard code Interpolation?
        # Use the Kalman filter predictor?
        # What to do? what to do?
        # Merge the 10 seconds data as a left join, i.e., outer join
        kiln_dataframe = kiln_dataframe.merge(
            kiln_dfs['data_10s'], on='timestamp', how='left')
        # leave it like this in the concatenator, any interpolation, do it in the
        # methid that separates the samples
        return kiln_dataframe

    def importDBs(self):
        # Get the 2 seconds data
        data_02seconds = dbImport.import_nseconds(self.path_02seconds,
                                                  "'data_2s'",
                                                  self.start_date,
                                                  self.end_date,
                                                  self.features_02seconds)
        # Get the 10 seconds data
        data_10seconds = dbImport.import_nseconds(self.path_10seconds,
                                                  "'data_10s'",
                                                  self.start_date,
                                                  self.end_date,
                                                  self.features_10seconds)
        # Get the transfos
        data_transfos = dbImport.import_transfos(
            self.path_transfos, self.start_date, self.end_date)
        # We are not currently using the 60 seconds, so keep going with what we have here
        # All is filtered by date. so return everithing in a dictionary
        kiln_dfs = {'data_02s': data_02seconds,
                    'data_10s': data_10seconds,
                    'transfos': data_transfos}
        return kiln_dfs

    @staticmethod
    def import_nseconds(path_to_db, table, start_date, end_date, features):
        # Function that accepts a path to a .db file,
        # a range to extract the data from,
        # and a set of strings to match the contents
        # 1. Create a connection
        connection = sqlite3.connect(path_to_db)
        # 2. Fetch the column names from the database
        db_cols = pd.read_sql_query(
            f'SELECT name FROM PRAGMA_TABLE_INFO({table})', connection)
        # 3. Match the columns that fir the list of features to include
        db_import_cols = db_cols.name[db_cols.name.str.contains(
            "|".join(features))]
        # 4. Now, the painfull part. Concatenation fo the huge string that gets the
        # required features from the selected time difference.
        # 4.1 string of features, it lacks a " at the beginning and at the end
        feature_str = '","'.join(db_import_cols)
        # 4.2 concatenate the string
        db_nseconds = pd.read_sql_query(
            f'SELECT "{feature_str}" FROM {table} WHERE ("timestamp" BETWEEN "{start_date}" AND "{end_date}") ORDER BY "TIMESTAMP"', connection)
        return db_nseconds

    @staticmethod
    def import_transfos(path_to_transfos, start_date, end_date):
        # The db I had no longer exists...
        # so, go back to csv NO! Now its a pickle
        # 1. load the pickle
        transfos = pd.DataFrame(pd.read_pickle(path_to_transfos))
        # transfos_csv = pd.DataFrame(pd.read_csv(path_to_transfos,
        #                                         encoding='utf-8',
        #                                         parse_dates=["Timestamp"],
        #                                         date_parser=pd.to_datetime))
        # Ok, this brings the complete file in the shape of a dictionary
        # I only need some variables in the required timeframe to
        # eventually do the synchronization. So... I can Dataframe it
        # immediately
        # 2. in the same line as one, i.e., pd. DataFrame
        # 3. Index only the necessary ones. It can be done in the same line...
        # Do it in anoter line and extract the dates and variables in the same slicing
        # I only need the Cycle - Number and the Combustion shaft
        # 4. Get just the necessary dates
        transfos = transfos.loc[(transfos["Timestamp"] >= pd.to_datetime(start_date)) & (
            transfos["Timestamp"] <= pd.to_datetime(end_date)), ["Cycle - Number", "Combustion shaft"]]
        # transfos_csv = transfos_csv.loc[(transfos_csv["Timestamp"]>=pd.to_datetime(start_date)) & (transfos_csv["Timestamp"]<=pd.to_datetime(end_date)),["Cycle - Number","Combustion shaft"]]
        return transfos


# if __name__ == "__main__":
#     import_params = dbImport()
#     print(import_params.akima_kiln_samples())
