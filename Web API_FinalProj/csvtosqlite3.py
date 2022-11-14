# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:57:51 2020

@author: ar54482
"""

import sqlite3
import pandas as pd

#Connect to the database
conn = sqlite3.connect('TestDB.db')  # You can create a new database by changing the name within the quotes
c = conn.cursor() # The database will be saved in the location where your 'py' file is saved

#Load the NS csv into the database
read_clients = pd.read_csv (r"C:\Users\ar54482\Desktop\DataCellShapeNS.csv")
read_clients.to_sql('ROOTHAIR', conn, if_exists='append', index = False) # Insert the values from the csv file into the table 'ROOTHAIR'

#Load the PS csv into the database
read_clients = pd.read_csv (r"C:\Users\ar54482\Desktop\DataCellShapeFinalPS.csv")
read_clients.to_sql('ROOTHAIR', conn, if_exists='append', index = False) # Insert the values from the csv file into the table 'ROOTHAIR' 

#Load the C csv into the database
read_clients = pd.read_csv (r"C:\Users\ar54482\Desktop\DataCellShapeFinalC.csv")
read_clients.to_sql('ROOTHAIR', conn, if_exists='append', index = False) # Insert the values from the csv file into the table 'ROOTHAIR'  

#View your table in the databse
cursor=conn.execute("""
    select * from ROOTHAIR
    """)

df=pd.DataFrame(cursor.fetchall(),columns=['ID','Area','Perimeter','Hausdorf','Frechet','Stress'])
print(df)

conn.close()