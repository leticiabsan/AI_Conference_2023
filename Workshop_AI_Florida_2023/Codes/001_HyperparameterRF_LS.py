# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 11:21:27 2022

@author: lsanto6
"""
#%% === Import libraries ===
import pandas as pd
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
import time
import winsound
import matplotlib.pyplot as plt
#from pprint import pprint

#%%==== 
main_dir =  os.getcwd()
Ex1 = pd.read_csv(main_dir+"\\Main\\Data\\FullData.csv")
#%%====
Ex1.dropna(subset=['YIELD_OBS'],inplace=True)
Ex2=Ex1[Ex1['YIELD_OBS']>0]
#%% === Constants ===
n= 50
#%%===
cols = list(Ex1) 
cols.insert(0, cols.pop(cols.index('YIELD_OBS')))
Ex1=Ex1.loc[:,cols]
df1_desc=Ex1.describe()
#%%===
Nancols=Ex2.columns[Ex2.isna().any()]
Ex2[Nancols]=Ex2[Nancols].fillna(Ex2.mean().iloc[0])
#%% === Get Dummies
Ex1feat = pd.get_dummies(Ex1)
Ex1num = Ex1feat.apply(np.float64)
#%%===
sc = StandardScaler()
Ex2 = pd.DataFrame(sc.fit_transform(Ex1num))
#%%====
X=Ex2.iloc[:,1:len(Ex2)].values
y=Ex2.iloc[:,0].values.flatten() #Skitlearn condition
#%% === Create empty dataframe to store results ===
results_1 = pd.DataFrame(columns=['rf_RMSE', "rf_MSE", 'lr_RMSE', 'lr_MSE', 'hyperparams', 'Time'])

#%% === 
for i in range(n):
    start=time.time()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
          
    #Random Forest
    regressor=RandomForestRegressor()
    model_hyp={'max_depth' : [6, 48], 
               'n_estimators': [100,1200],
               "max_features": ['sqrt', 'log2']}
    
    rf_opt=GridSearchCV(regressor,model_hyp)
    rf_opt.fit(X_train,y_train)
    y_pred_rf=rf_opt.predict(X_test)
    RMSE_rf=np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf))
    MSE_rf=metrics.mean_squared_error(y_test,y_pred_rf)
    hyp = rf_opt.best_params_ 
    
    #Linear Regression
    regr=linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    y_pred_lr=regr.predict(X_test)
    RMSE_lr=np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr))
    MSE_lr=metrics.mean_squared_error(y_test, y_pred_lr)
     
    
    #Append other results
    end = time.time()
    tm = end-start
    results_1 = results_1.append({'rf_RMSE': RMSE_rf, "rf_MSE": MSE_rf,
                                  'lr_RMSE': RMSE_lr, "lr_MSE": MSE_lr,
                                  "hyperparams": hyp, 'Time': tm}, ignore_index=True)
            
winsound.Beep(440,500)


#%%===
results_MSE = results_1[['rf_MSE', 'lr_MSE']]
results_TimeHyp = results_1[['hyperparams', 'Time']]

results_longMSE = pd.melt(results_MSE, var_name="Method", value_name="MSE")


results_longMSE.to_csv(main_dir+'\Main\Results\ResultsMSE_OptimizedRF.csv')
results_TimeHyp.to_csv(main_dir+'\Main\Results\ResultsHyperparameters_OptimizedRF.csv', index=False)



#%%
fig, ax = plt.subplots()
# 
ax.boxplot(results_MSE)
# 
ax.set_title('Side by Side Boxplot of MSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('Mean Square Errors')
xticklabels=['Random Forest','Simple Regression']

ax.set_xticklabels(xticklabels)
# 
ax.yaxis.grid(True)
# 
plt.savefig(main_dir+'\Main\Results\Side_by_Side_OptimizedRF.png')