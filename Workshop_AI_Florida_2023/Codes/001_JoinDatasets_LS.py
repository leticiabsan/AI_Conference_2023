# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:06:07 2023

@author: Leticia
"""
#%% === Import Datasets ===
import pandas as pd
import os
#%% === Set the directory containing the data ===
# Click on the folder icon on the top right and select the folder AI_Florida2023 - wherever you saved it on your pc
path =  os.getcwd()
main_dir=(path+"\Main\Data")
#%% === 
list_CSV=[]
#%%
for subdir in os.listdir(main_dir): 
    sub_dir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(sub_dir_path):
       
        for filename in os.listdir(sub_dir_path):
            if filename.endswith('.csv'):
                csv_path = os.path.join(sub_dir_path, filename)
                csv_data = pd.read_csv(csv_path)
                list_CSV.append(csv_data)

               
# Merge the dataframes on the common column 'ID'
merged = pd.concat(list_CSV, axis=1, ignore_index=False)
merged = merged.loc[:,~merged.columns.duplicated()]
#%% === Save the combined CSV data to a new file ===
merged.to_csv(path+"\Main\Data\FullData.csv", index=False)