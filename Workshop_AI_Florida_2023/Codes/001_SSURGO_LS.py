# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:53:48 2023

@author: lsanto6
"""
#%% Import libraries and set up the working directory 
# Click on the folder icon on the top right and select the folder AI_Florida2023 - wherever you saved it on your pc
import pandas as pd
import geopandas as gpd
import os
path =  os.getcwd()
#%% Read files
# Read the CSV file into a pandas dataframe
df = pd.read_csv(path+'\Main\Data\SSURGO\Data\SSURGO_LatLong.csv')

# Create a geopandas dataframe with the lat and long columns, from the csv
gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['long'], df['lat']))

# Read the shapefiles 
gdf_Catahoula = gpd.read_file(path+'\Main\Data\SSURGO\Data\wss_aoi_2023-04-13_18-41-04\spatial\soilmu_a_aoi.shp', index_col=False)
gdf_RRS = gpd.read_file(path+'\Main\Data\SSURGO\Data\wss_aoi_2023-04-13_18-46-52\spatial\soilmu_a_aoi.shp', index_col=False)
gdf_BenHur = gpd.read_file(path+'\Main\Data\SSURGO\Data\wss_aoi_2023-04-13_19-34-14\spatial\soilmu_a_aoi.shp', index_col=False)

#%%Visualization
gdf_points.plot() #Visualize the points
gdf_Catahoula.plot() # Visualize the polygons downloaded from SSURGO Website
gdf_RRS.plot()
gdf_BenHur.plot()

#%% Overlay polygons, so you will have them all together
# Merge the polygons using the concat function from pandas
merged = pd.concat([gdf_Catahoula, gdf_RRS, gdf_BenHur], ignore_index=True)

#%% Intersect the points and the maps (polygons) 
gdf_merged = gpd.sjoin(gdf_points, merged, how='left', op='within')
column_names = gdf_merged.columns.tolist()
gdf_merged = gdf_merged.drop(['LOCATION', 'lat', 'long', 'geometry','index_right', 'AREASYMBOL','SPATIALVER','MUKEY'], axis=1)


#%% Save the modified dataframe to a new CSV file
gdf_merged.to_csv(path+"\Main\Data\SSURGO\SSURGO_04_05_2023.csv")
