# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:05:10 2019

@author: MMOHTASHIM
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use("ggplot")

class Support_Vector_Machine(object):
    def __init___(self,visulization=True):
        self.visulization=visulization
        self.colors={1:"r",-1:"b"}
        if self.visulization:
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(1,1,1)
    ###traing the data to find w and b
    def fit(self,data):
       self.data=data
        ####{||w||:[w,b]} a dictionary which store for every modulus value of w , a associate vector w and b
       opt_dict={}
        
        
        ##These transforms are what we use to apply to vector w
        ##each time we step in order to know ever possible direction of a vector w and its 
        ##associate b value whose value is affected by direction and store the highest b value, as modulus of w doesn't account 
        #for direction,remeber in vector direction matters
       transforms=[[1,1],[-1,1],[-1,-1],[1,-1]]
        
        
        
        
    
       all_data=[]
        ###this three loop takes all features of the associated class yi and make,
        ##a new list of these features and than take the max and min value associated with the 
        ### this new list of feature and these max and min values are to be used for further convex optimization
       for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
       self.max_feature_value=max(all_data)
       self.min_feature_value=min(all_data)
       all_data=None
       step_size=[self.max_feature_value*0.1,
                  self.max_feature_value*0.01,
                  ##POINT OF EXPENSE
                  self.max_feature_value*0.001]
       ###extremely expensive-b does not need to take precise step 
       b_range_multiple=5
       
       #we dont need to take as small of steps
       #wit b as we do w
       b_multiple=5
       ###the first value of w and remeber to simplify things,we assume each element of vector w
       ### to be same
       latest_optimum=self.max_feature_value*10
       
       for step in step_size:
           ####remeber to simplify things,we assume each element of vector w
       ### to be same
           w=np.array([latest_optimum,latest_optimum])
           
           
           #we can do this because convex alogrithm
           optimized=False
           while not optimized:
               ####setting a range for b 
              for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                 self.max_feature_value*b_range_multiple,
                                 step*b_multiple):
                  for transformation in transforms:
                      ##applying the different transformation to account for difeerent direction(w_t)
                      w_t=w*transformation
                      found_option=True
                      #weakest link in the SVM fundamentally
                      #SMO attempt to fix this a bit
                      ##Running the data on all points is costly-svm weakness
                      ##yi(xi.w+b)>=1(constraint)
                      for i in self.data:
                          for xi in self.data[i]:
                              yi=i
                              ###this condition check even if one point in our data doesnt fit the constraint with the give w vector
                              if not yi*(np.dot(w_t,xi)+b)>=1:
                                  found_option=False
                      #if w satisfies the constraint
                      if found_option:
                          opt_dict[np.linalg.norm(w_t)]=[w_t,b]
              if w[0]<0:
                optimized=True
                print("optimized a step.")
              else:
                w=w-step
           #taking the smallest modulus w and taking new starting new point
           norms=sorted([n for n in opt_dict])
           opt_choice=opt_dict[norms[0]]
           self.w=opt_choice[0]
           self.b=opt_choice[1]
           latest_optimum=opt_choice[0][0]+step*2
    def predict(self,features):
        ###sign(x.w+b) whatever the sign of the equation is
        classification=np.sign(np.dot(np.array(features),self.w)+self.b)
            
        return classification     






data_dict={-1:np.array([[1,7],
                        [2,8],
                        [3,8]]),
            1:np.array([[5,1],[6,-1],[7,3]])}
