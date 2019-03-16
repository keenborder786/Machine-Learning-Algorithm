# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:43:39 2019

@author: MMOHTASHIM
"""

#K-model from scratch:
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import random 
import pandas as pd
###creating a dataset with labels and features
dataset={"k":[[1,2],[2,3],[3,1]],"r":[[6,5],[7,7],[8,6]]}
##feature to be classified
new_feature=[5,7]
####Looping over to scatter the plot
#for i in dataset:
#    for ii in dataset[i]:
#        [plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]]
###More pythontic way
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_feature[0],new_feature[1])
#plt.show()
###k alogorithm
def k_nearest_neighbours(data,predict,k=3):
    if len(data) >=k:
        warnings.warn("K is set to a value less than total voting groups!")
    distances=[]
    for group in data:
        for features in data[group]:
#            euclidean_distance=np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) or better:
             ###numpy fomula
             euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
             #####Making the euclidean distance list to sort later
             distances.append([euclidean_distance,group])
    ###calculating votes to help us classify-lowest distance
    votes=[i[1] for i in sorted(distances)[:k]]
    votes_result=Counter(votes).most_common(1)[0][0]
    ###confidence measure how confident our classifer is about one single point in labelling that point-that is what porppotion of votes were infavour
    confidence=Counter(votes).most_common(1)[0][1]/k
    return votes_result,confidence

#result=k_nearest_neighbours(dataset,new_feature,k=3)
#print(result)
###showing the result,color is already k an r as variables.
#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_feature[0],new_feature[1],color=result)
#plt.show()
######################Comparing our model vs sklearn model
#df=pd.read_csv("breast-cancer-wisconsin.data.txt")
#df.replace("?",-9999,inplace=True)
#df.drop(["id"],1,inplace=True)
#full_data=df.astype(float).values.tolist()
######shuffling the inner lists of full_data
#random.shuffle(full_data)
#
#########dividing the full data into train data and test data
#test_size=0.4
#train_set= {2:[],4:[]}
#test_set={2:[],4:[]}
#train_data=full_data[:-int(test_size*len(full_data))]
#test_data=full_data[-int(test_size*len(full_data)):]
#
#for i in train_data:
#    ####associating the datas to classifiers in this case 2 or 4
#    train_set[i[-1]].append(i[:-1])
#for i in test_data:
#    ####associating the datas to classifiers in this case 2 or 4
#    test_set[i[-1]].append(i[:-1])
#correct=0
#total=0
#for group in test_set:
#    for data in test_set[group]:
#        vote,confidence=k_nearest_neighbours(train_set,data,k=5)
#        if group == vote:
#            correct+=1
#        else:
#            print(confidence)
#        total+=1
#        
#print("Accuracy: ",correct/total)
#########################copying the whole algorith down again to judge the accuracy in a numbe of trials:
accuracies=[]
n=25
for i in range(n):
    df=pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace("?",-9999,inplace=True)
    df.drop(["id"],1,inplace=True)
    full_data=df.astype(float).values.tolist()
    #####shuffling the inner lists of full_data
    random.shuffle(full_data)
    
    ########dividing the full data into train data and test data
    test_size=0.4
    train_set= {2:[],4:[]}
    test_set={2:[],4:[]}
    train_data=full_data[:-int(test_size*len(full_data))]
    test_data=full_data[-int(test_size*len(full_data)):]
    
    for i in train_data:
        ####associating the datas to classifiers in this case 2 or 4
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        ####associating the datas to classifiers in this case 2 or 4
        test_set[i[-1]].append(i[:-1])
    correct=0
    total=0
    for group in test_set:
        for data in test_set[group]:
            vote,confidence=k_nearest_neighbours(train_set,data,k=5)
            if group == vote:
                correct+=1
            total+=1
    accuracies.append(correct/total)
print("overall_accuracy(our algorithm) for", n ," steps = ", sum(accuracies)/len(accuracies))
##############finally getting the sklearn algorithm and comparing it with our overall accuracy for a specific number of steps:
accuracies_2=[]
for i in range(n):
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    df=pd.read_csv("breast-cancer-wisconsin.data.txt")
    ##fill missing data,-9999 will be treated as outlier in our algorithm and dont,
    #lose rest of the data
    df.replace("?",-9999,inplace=True)    
    ###check for any useless data and drop it 
    df.drop(["id"],1,inplace=True)
    #### X are the features and y is the label
    X=np.array(df.drop(["class"],1))
    y=np.array(df["class"])    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    #####Using the classifer
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    #Remeber the difference between accuracy and confidecnce
    accuracy=clf.score(X_test,y_test)
    accuracies_2.append(accuracy)
print("overall_accuracy(sk-learn alogorithm) for", n ," steps = ", sum(accuracies_2)/len(accuracies_2))



