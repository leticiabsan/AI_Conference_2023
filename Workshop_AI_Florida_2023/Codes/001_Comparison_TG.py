# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:09:35 2021

@author: genti
"""
#%%====
import pandas as pd
import os
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt

#%%==== 
main_dir =  os.getcwd()
Ex1 = pd.read_csv(main_dir+"\\Main\\Data\\FullData.csv")
Ex1.columns
#%%====
Ex1.dropna(subset=['YIELD_OBS'],inplace=True)
Ex2=Ex1[Ex1['YIELD_OBS']>0]
#%%====
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('YIELD_OBS')))
Ex2=Ex2.loc[:,cols]
df1=Ex2.describe()
df1.to_csv(main_dir+'/Main/Results/Data_Description.csv')
#%%==== 
Nancols=Ex2.columns[Ex2.isna().any()]
Ex2[Nancols]=Ex2[Nancols].fillna(Ex2.mean().iloc[0])
#%%==== 
Ex3 = pd.get_dummies(Ex2)
#%%==== 
X=Ex3.iloc[:,1:len(Ex3.columns)].values
y=Ex3.iloc[:,0].values.flatten()
#%%==== 
rf_errors = []
regr_errors = []
nn_errors = []
#%%==== 
for i in range(10):
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    #
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)  

    #
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)  
    regressor.fit(X_train, y_train)  
    RF_pred = regressor.predict(X_test)  
    #
    a=metrics.mean_squared_error(y_test, RF_pred)
    rf_errors.append(a)
    
    #
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    LR_pred=regr.predict(X_test)
    #
    b=metrics.mean_squared_error(y_test, LR_pred)
    regr_errors.append(b)

    #
    nn=models.Sequential()
    nn.add(layers.Dense(80, activation='relu', input_shape=[X_train.shape[1]]))
    nn.add(layers.Dense(40, activation='relu'))
    nn.add(layers.Dense(20, activation='relu'))
    nn.add(layers.Dense(10, activation='relu'))
    nn.add(layers.Dense(1))
    nn.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    nn.fit(X_train, y_train, epochs=10, batch_size=5)
    NN_pred = nn.predict(X_test)
    #
    c=metrics.mean_squared_error(y_test, NN_pred)
    nn_errors.append(c)
    
    print("Finished Itteration",i)
#%%==== 
Result_errors1=pd.DataFrame({'Random_Forest':rf_errors,"Simple_Regression":regr_errors,'Neural_Network':nn_errors})
#Result_errors2=Result_errors1[Result_errors1["Simple_Regression"]<600]
#Result_errors3=Result_errors2.sample(n=6)
Result_errors1.to_csv(main_dir+'\Main\Results\Errors.csv')
#%%==== 
fig, ax = plt.subplots()
# 
ax.boxplot(Result_errors1)
# 
ax.set_title('Side by Side Boxplot of MSE for different Models')
ax.set_xlabel('Predictive Models')
ax.set_ylabel('Root Mean Square Errors')
xticklabels=['Random Forest','Simple Regression','Neural Networks']

ax.set_xticklabels(xticklabels)
# 
ax.yaxis.grid(True)
# 
plt.savefig(main_dir+'\Main\Results\Side_by_Side.png')
#%%====
Res=Result_errors1.describe()
Res.to_csv(main_dir+'\Main\Results\Error_description.csv')
