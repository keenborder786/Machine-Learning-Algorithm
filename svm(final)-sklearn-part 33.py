# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 23:51:30 2019

@author: MMOHTASHIM
"""
import numpy as np
from sklearn import svm,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace("?",-9999,inplace=True)
df.drop(["id"],1,inplace=True)


X=np.array(df.drop(["class"],1))
y=np.array(df["class"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


clf=svm.SVC(gamma="auto",kernel="rbf")
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)