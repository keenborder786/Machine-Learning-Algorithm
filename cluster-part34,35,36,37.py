# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:06:06 2019

@author: MMOHTASHIM
"""

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype

#####basic visulisation of k-means
X=np.array([[1,2],
           [1.5,1.8],
           [5,8],
           [8,8],
           [1,0.6],
           [9,11]])
plt.scatter(X[:,0],X[:,1],s=150)



clf=KMeans(n_clusters=2)
clf.fit(X)
centroids=clf.cluster_centers_
labels=clf.labels_
colors=["g.","r.","c.","b.","k.","o."]
print(centroids)
print(labels)
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=25)
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=150)
plt.show()
##################################################################
###Analysing Titanic dataset through K-means
df=pd.read_excel("titanic.xls")
df.drop(["body","name"],1,inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0,inplace=True)
c=df["age"].values.tolist()
###in order to convert text data to useable numeric data
def handle_numeric_data(df):
    columns=df.columns.values
#    dtypes=dict(df.dtypes)
    for column in columns:
        text_digit_vals={}
#        dtype=dtypes[column]
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            columns_contents=df[column].values.tolist()

            unique_element=set(columns_contents)
            x=0
            for unique in unique_element:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
                    
            df[column]=list(map(convert_to_int,df[column]))
    return df
df=handle_numeric_data(df)
####################################################################

#df.drop(["sex","boat"],1,inplace=True)
X=np.array(df.drop(["survived"],1),dtype=float)
X=preprocessing.scale(X)
y=np.array(df["survived"])


clf=KMeans(n_clusters=2)
clf.fit(X)

########In unsupervised learning we do not have labels so we are going to use only X to fit the data
########then KMeans would label the data into two groups and to check accuracy of that labelling we will see
#######what was the prediction of the KMeans and compare with binary representation of our y.
correct=0
labels=clf.labels_###use this
print(labels)
for i in range(len(X)):
    predict_me=np.array(X[i],dtype=float)
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)###what does the label given by classifer
    if prediction[0]==y[i]:####did it label correctly
        correct+=1
print(correct/len(X))    

