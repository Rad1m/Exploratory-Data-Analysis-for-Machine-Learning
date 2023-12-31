# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:25:12 2023

@author: simanekr
"""

# SQL Data Imports
import pandas as pd
import sqlite3 as sq3
import pandas.io.sql as pds
import numpy as np


### BEGIN SOLUTION
# Create a variable, `path`, containing the path to the `baseball.db` contained in `resources/`
path = 'data/baseball.db'

# Create a connection, `con`, that is connected to database at `path`
con = sq3.Connection(path)

# Create a variable, `query`, containing a SQL query which reads in all data from the `` table

query = """
SELECT *
    FROM allstarfull
    ;
"""

allstar_observations = pd.read_sql(query, con)

#print(allstar_observations)

# Create a variable, tables, which reads in all data from the table sqlite_master
all_tables = pd.read_sql('SELECT * FROM sqlite_master', con)
#print(all_tables)

# Pretend that you were interesting in creating a new baseball hall of fame. Join and analyze the tables to evaluate the top 3 all time best baseball players
best_query = """
SELECT playerID, sum(GP) AS num_games_played, AVG(startingPos) AS avg_starting_position
    FROM allstarfull
    GROUP BY playerID
    ORDER BY num_games_played DESC, avg_starting_position ASC
    LIMIT 50
"""
best = pd.read_sql(best_query, con)
### END SOLUTION

# get avg_starting_position and convert it from SQL to Python Array
data =np.array(list(best["avg_starting_position"]))
data= data[~pd.isnull(data)]

# calculate the interquartile range
q25, q50, q75 = np.percentile(data, [25, 50, 75])
iqr = q75 - q25

# calculate the min/max limits to be considered an outlier
min = q25 - 1.5*(iqr)
max = q75 + 1.5*(iqr)

print ("Min: ", min,"\n","Q25: ", q25, "\n","Q50: ", q50,"\n","Q75: ", q75,"\n","Max: ", max)