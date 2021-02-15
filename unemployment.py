# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:44:03 2020

@author: Tejaswiny
"""

# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import math
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# IMPORTING THE DATASET
dataset = pd.read_excel(r'C:\Users\ASUS\Desktop\Project\DataSet.xlsx')

# CHECKING THE VALIDITY OF DATASET
print(dataset.head())

# CHANGING THE NAME OF THE COLUMN PLACEHOLDERS
dataset = dataset.rename(columns={'Value':'UNEMPLOYMENT RATE'})
dataset = dataset.rename(columns={'Year':'YEAR NUMBER'})
dataset = dataset.rename(columns={'Period':'MONTH NUMBER'})
print(dataset.head())

# CREATE A NEW TABLE FOR EACH YEAR AND ITS AVG RATE.
info = pd.DataFrame(dataset['YEAR NUMBER'].unique(), columns=['YEAR NUMBER'])

# FIND THE MEAN UNEMPLOYMENT RATE OF EACH YEAR.
sum=0
avg=[]
n=0
for x in range(len(info)):
    for y in range(n,len(dataset)):
        if(dataset['YEAR NUMBER'][y] == info['YEAR NUMBER'][x]):
            sum += dataset['UNEMPLOYMENT RATE'][y]
        else:
            avg.append(sum/12)
            n=y
            sum=0
            break
        if(y == 839): # y will never reach 840, so without this condition, the else condition above will not be activate
            avg.append((sum/12))
            
    
# COMBINE THE DATA.
info['UNEMPLOYMENT RATE'] = pd.DataFrame(avg, columns=['UNEMPLOYMENT RATE'])

# ROUNDING OFF THE DATA.
info['UNEMPLOYMENT RATE'] = info['UNEMPLOYMENT RATE'].round(2)

# CHECKING.
print(info.head())

# SHOW THE ACTUAL GRAPH
fig,ax = plt.subplots(figsize=(15,5))
ax.scatter(info['YEAR NUMBER'], info['UNEMPLOYMENT RATE'],color='blue')
ax.plot(info['YEAR NUMBER'], info['UNEMPLOYMENT RATE'],color='orange')

# MORE DETAILING FOR THE YEAR.
ax.locator_params(nbins=15, axis='x')

# ADDING DETAILS.
plt.title('UNEMPLOYMENT RATE FROM 1948 TO 2017',fontsize=20,color = 'green')
plt.xlabel( 'YEAR NUMBER',fontsize = 20,color = 'red')
plt.ylabel( 'RATE OF UNEMPLOYMENT',fontsize = 20,color = 'red')
plt.show()

#   WE NNED TO LOG TRANSORM THE 'y' VARIABLE TO TRY TO CONVERT NON-STATIONARY DATA TO STATIONARY.  
info['UNEMPLOYMENT RATE'] = np.log(info['UNEMPLOYMENT RATE'])


# PREPARE THE TRAINING DATA.
d_set = info['UNEMPLOYMENT RATE'].values
train_set = d_set[:50]
X_train = []
Y_train = []
for i in range(30, len(train_set)):
    X_train.append(train_set[i-30:i])
    Y_train.append(train_set[i])
    
# PREPARE THE TEST SET. 
test_set = d_set[20:]
X_test = []
Y_test = d_set[50:]
for i in range(30, 50):
    X_test.append(train_set[i-30:i]) 
    
# LINEAR REGRESSION
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
pred_val = regressor.predict(X_test)

# REVERSE THE VALUES FROM LOG 
for i in range(20):
    Y_test[i] = math.exp(Y_test[i])
    pred_val[i] = math.exp(pred_val[i])

# last 20 years
L20y = info['YEAR NUMBER'][50:]

# PLOTTING THE GRAPH.
fig,ax = plt.subplots(figsize=(15,7))
ax = fig.add_axes([0.0, 0.0, .9, .9], polar=True)
one, = ax.plot(L20y, Y_test, color='orange',lw=2,ls='-',marker='o',markersize=8,markerfacecolor="purple")
two, = ax.plot(L20y, pred_val, color='purple',lw=2)
two.set_dashes([5,10,15,10])
plt.legend([one,two],['Original','Predicted'])
plt.xlabel( 'YEAR NUMBER',fontsize = 20,color = 'green')
plt.ylabel( 'RATE OF UNEMPLOYMENT',fontsize = 20,color = 'green')
plt.title('LINEAR REGRESSION',fontsize=20,color='purple')
ax.locator_params(nbins=20, axis='x')

# RANDOM FOREST
rforest = RandomForestRegressor(n_jobs=100)
rforest.fit(X_train, Y_train)
pred_rforest_val = rforest.predict(X_test)

for i in range(20):
    pred_rforest_val[i] = math.exp(pred_rforest_val[i])

# PLOTTING THE GRAPH   
fig,ax = plt.subplots(figsize=(15,7))
one, = ax.plot(L20y, Y_test, color='orange',lw=2,ls='-',marker='o',markersize=8,markerfacecolor="purple")
two, = ax.plot(L20y, pred_rforest_val, color='purple',lw=2,ls='-.')
plt.legend([one,two],['Original','Predicted'])
plt.xlabel( 'YEAR NUMBER',fontsize = 20,color = 'green')
plt.ylabel( 'RATE OF UNEMPLOYMENT',fontsize = 20,color = 'green')
plt.title('RANDOM FOREST',fontsize=20,color='orange')
ax.locator_params(nbins=20, axis='x')

# KNN
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, Y_train)
pred_knn_val= knn.predict(X_test)
for i in range(20):
    pred_knn_val[i] = math.exp(pred_knn_val[i])

# PLOTTING THE GRAPH
fig,ax = plt.subplots(figsize=(15,7))
#ax = fig.add_axes([0.0, 0.0, .9, .9], polar=True)
one, = ax.plot(L20y, Y_test, color='orange',lw=2,ls='-',marker='o',markersize=8,markerfacecolor="purple")
two, = ax.plot(L20y, pred_knn_val, color='purple',lw=2,ls=':')
plt.legend([one,two],['original','predicted'])
plt.xlabel( 'YEAR NUMBER',fontsize = 20,color = 'green')
plt.ylabel( 'RATE OF UNEMPLOYMENT',fontsize = 20,color = 'green')
plt.title('KNN',fontsize=20,color='black')
ax.locator_params(nbins=20, axis='x')

# OVERALL GRAPH REPRESENTATION

fig,ax = plt.subplots(figsize=(15,7))
#ax = fig.add_axes([0.0, 0.0, 0.9, 0.9], polar=True)
a, = ax.plot(L20y, Y_test, color='orange',lw=2,ls='-',marker='o',markersize=8,markerfacecolor="purple")
b, = ax.plot(L20y, pred_knn_val, color='brown',lw =2,ls=':')
c, = ax.plot(L20y, pred_rforest_val, color='blue',lw =2,ls='-.')
plt.xlabel( 'YEAR NUMBER',fontsize = 20,color = 'green')
plt.ylabel( 'RATE OF UNEMPLOYMENT',fontsize = 20,color = 'green')
plt.legend([a,b,c],['Original', 'KNN','Randon Forest'])
plt.title('OVERALL COMPARISON',fontsize=20,color = 'red')




