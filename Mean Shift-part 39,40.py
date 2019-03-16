# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:01:02 2019

@author: MMOHTASHIM
"""
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
############Basic Visulisation of Mean Shift
#centers = [[1,1,1],[5,5,5],[3,10,10]]
#
#X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1.5)
#
#ms = MeanShift()
#ms.fit(X)
#labels = ms.labels_
#cluster_centers = ms.cluster_centers_
#
#print(cluster_centers)
#n_clusters_ = len(np.unique(labels))
#print("Number of estimated clusters:", n_clusters_)
#
#colors = 10*['r','g','b','c','k','y','m']
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for i in range(len(X)):
#    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')
#
#ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
#            marker="x",color='k', s=150, linewidths = 5, zorder=10)

######################################################################
# -*- coding: utf-8 -*-
#############Mean Shift on Titanic Dataset
df = pd.read_excel('titanic.xls')
orginal_df=pd.DataFrame.copy(df)


df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
print(df.head())
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head())

# add/remove features just to see impact they have.
df.drop(['ticket','home.dest'], 1, inplace=True)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

clf = MeanShift()##change this
clf.fit(X)


labels=clf.labels_
cluster_centers=clf.cluster_centers_

orginal_df["cluster_group"]=np.nan
##########in order to check survial for each individual cluster formed by MeanShift and check accuracy of clustes formed
for i in range(len(X)):
    orginal_df["cluster_group"].iloc[i]=labels[i]
n_clusters_=len(np.unique(labels))
survival_rates={}###to see survival rate for different classes
for i in range(n_clusters_):
    temp_df=orginal_df[(orginal_df["cluster_group"]==float(i))]###make a temp dataframe for each cluster
#    print(temp_df)
    survival_cluster=temp_df[(temp_df["survived"]==1)]
    survival_rate=len(survival_cluster)/len(temp_df)
    survival_rates[i]=survival_rate
#print(orginal_df[(orginal_df["cluster_group"]==2)])
###Now you can use df.describe() to analyse the data for different classes
    