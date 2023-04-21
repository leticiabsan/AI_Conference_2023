# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:45:51 2022

@author: lsanto6
"""
#%% Libraries and Data
import pandas as pd
import seaborn as sns
import os
import pylab as plt
os.getcwd()
#%%
main_dir = os.getcwd()
df=pd.read_csv(main_dir+'\Main\Results\ResultsHyperparameters_OptimizedRF.csv')
list(df)
df.dtypes
#%% Hyperparameters mode and Amount of time
df["hyperparams"] = df["hyperparams"].apply(lambda x : dict(eval(x)) )
df1 = df["hyperparams"].apply(pd.Series)
df1 = pd.concat([df, df1], axis=1).drop('hyperparams', axis=1)

df2=df.describe(include='all')
df3=df1.loc[:,['max_depth','max_features','n_estimators']].astype('category')
df3.mode(axis=0)
df3.loc[:,"n_estimators"].mode() # Mode of a specific column

#Checking by column
df3['n_estimators'].value_counts()
df3['max_features'].value_counts()
df3['max_depth'].value_counts()

#Time counting
df['Time'].describe()
df['Time'].sum(axis=0, skipna=True) # 29996.34s or 8.33 hrs

#%% Plotting Results (Check if it's interesting for paper results)
sns.set(style='ticks', palette = "colorblind", font_scale=1.5)
fig, ax = plt.subplots(1,3, figsize=(9,6))
fig.subplots_adjust(hspace=0.45, wspace=0.45)
ax1=sns.countplot(df3['n_estimators'], order = df3['n_estimators'].value_counts().index, ax=ax[0])
ax2=sns.countplot(df3['max_features'], order = df3['max_features'].value_counts().index, ax=ax[1])
ax3=sns.countplot(df3['max_depth'], order = df3['max_depth'].value_counts().index, ax=ax[2])
plt.suptitle('Counts for Hyperparameters')
#ax1.title.set_text("First")
#ax1.set(xlabel='common xlabel', ylabel='common ylabel')
fig.savefig(main_dir+'\Main\Results\ResultsHyperparameters_OptimizedRF.png')

