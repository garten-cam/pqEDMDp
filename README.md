# Introduction

Data management in sqlite

1. The initial script imports sqlite3, starts a connection but then does a pandas query to get the data. It uses the sql hability from pandas. There must be a way of doing everything within the database and not pandas.
1.0. I forgot a lot of things before database management
1.0.1. Make a script that takes all the csvs from a directory and adds them in a single database. (Done fot the individual files, but not as a concatenation in a single database)
1.0.2. Make a script that does the same with all the databases in a directory, concatenate in a single file.
1.1. Generate connection using sqlite3. What is the best option? Create a .sql file to concatenate everything in a db? Or call everything from the python script and do it in python?

2. Ok, the scripting in databases is giving me a lot of problems. I will import the data from the database into matlab and do the analysis. I am running out of time and that can be dealt later.
2.1 Import everything from .db files 
2.2 do the same data transformation and selection (should be a lot of copy and paste)
2.2.1 The are is a problem with the import, the trasfos file and the "seconds" files need to have a specific data range because otherwise the import becomes very computationally expensive. Also, for example in the transfos file, the necessary information is a couple of columns. So, I will work as before, getting the complete database imported, and later on, join using matlab.  IMPORTANT some db expert can help me do all these joins directly in SQLite

3. 
