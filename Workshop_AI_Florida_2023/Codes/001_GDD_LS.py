# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:29:22 2023

@author: lsanto6
"""
#%% Libraries and Dataset
import pandas as pd
import os
import seaborn as sns
#%% Define base temperature and input data file
# Click on the folder icon on the top right and select the folder AI_Florida2023 - wherever you saved it on your pc
Tbase = 50 # Fahrenheit
path =  os.getcwd()
df = pd.read_csv(path+"\\Main\\Data\\Weather\\Raw\\Weather.csv")
ID_collumn = pd.read_csv(path+"\\Main\\Data\\Weather\\Raw\\PhenotypicDates_03_23_2023.csv")
ID_collumn = pd.DataFrame(ID_collumn[["ID", "LOCATION"]])


#%% Calculate daily GDD values
df["GDD"] = (df["MAXTEMP"] + df["MINTEMP"])/2 - Tbase
df["GDD"] = df["GDD"].apply(lambda x: 0 if x < 0 else x) # set negative values to 0

# calculate cumulative GDD values for the period
df['accumulated_GDD'] = df.groupby('LOCATION')['GDD'].transform('cumsum')
df['GDD_cum']= df.groupby('LOCATION')['GDD'].transform("sum")
df1=df.drop(['Date', 'MINTEMP', 'MAXTEMP', 'GDD','accumulated_GDD'], axis=1)
df_GDD=df1.mode()
merged = pd.merge(ID_collumn, df_GDD, on="LOCATION")
#%% Visualization
sns.lineplot(data=df, x='Date', y="accumulated_GDD", hue='LOCATION')

#%% Saving file
merged.to_csv(path+"\Main\Data\Weather\Weather_03_23_2023.csv", index=False)


