# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:38:21 2020

@author: ar54482
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect('TestDB.db')  # You can create a new database by changing the name within the quotes
c = conn.cursor() # The database will be saved in the location where your 'py' file is saved

c.execute("DROP TABLE If exists ROOTHAIR")

# Create table - CLIENTS
c.execute('''CREATE TABLE ROOTHAIR
             ([ID] text,[Area] REAL, [Perimeter] REAL, [Hausdorf] REAL, [Frechet] REAL, [Stress] text)''')

conn.commit()

read_clients = pd.read_csv (r"C:\Users\ar54482\Desktop\DataCellShapeFre.csv")
read_clients.to_sql('ROOTHAIR', conn, if_exists='append', index = False) # Insert the values from the csv file into the table 'CLIENTS' 

