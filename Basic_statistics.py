# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:44:25 2023

@author: simanekr
"""

import numpy as np
import pandas as pd
import sqlite3 as sq3
import pandas.io.sql as pds

# Create a variable, `path`, containing the path to the `baseball.db` contained in `resources/`
path = 'data/baseball.db'

# data range
data = [0, 10, 20, 30, 40, 50, 80]

# calculate the interquartile range
q25, q50, q75 = np.percentile(data, [25, 50, 75])