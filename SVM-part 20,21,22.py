# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:21:35 2019

@author: MMOHTASHIM
"""

import numpy as np
from sklearn import preprocessing,neighbors,svm
from sklearn.model_selection import cross_validate,train_test_split
import pandas as pd
import pickle
df=pd.read_csv("breast-cancer-wisconsin.data.txt")
##fill missing data,-9999 will be treated as outlier in our algorithm and dont,
#lose rest of the data
df.replace("?",-9999,inplace=True)

###check for any useless data and drop it 
df.drop(["id"],1,inplace=True)
#### X are the features and y is the label
X=np.array(df.drop(["class"],1))
print(X)
y=np.array(df["class"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#####Using the classifer
clf=svm.SVC()
clf.fit(X_train,y_train)
###Saving the classifer
with open("K_model","wb") as f:
    pickle.dump(clf,f)
#Remeber the difference between accuracy and confidecnce
accuracy=clf.score(X_test,y_test)
print(accuracy)
####make prediction
predict_X=np.array([[4,2,1,1,1,2,3,2,1],[4,2,2,1,2,2,3,2,1]])
example_measures=np.array(predict_X)
print(example_measures)
###To make the array shape that sklearn understands and matches the the X features
predict=clf.predict(example_measures.reshape(len(example_measures),-1))
print(predict)



###########the part 23,24 was theory