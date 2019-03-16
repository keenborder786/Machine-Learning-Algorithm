# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:00:48 2019

@author: MMOHTASHIM
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use("ggplot")

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
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
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*', c=self.colors[classification])
        return classification     
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        def hyperplane(x,w,b,v):
            ###v=x.w+b
            ###the hyperplane function shows the support vector plannes and boudrt decision so:
            ###positive support vector(psv)=1
            ###nsv=-1
            ###decision boundary=0,want to find a plane with these associated v values and show them
            #hyperplane v=x.w+b
            ##x,y is an unknown point on the hyperplane
    #        x_v and w_v are the vector
    #        x_v= [x,y]
    #        x_v.w_v+b =1 for postive sv
    ## this helps to find the value of y where value of hyperplance is 1
            return (-w[0]*x-b+v)/w[1]
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        #(w.x+b)=1
        #positive support vector hyperplane
        psv1=hyperplane(hyp_x_min,self.w,self.b,1)
        ##psv1 is going to be scalar value not vector and its going to be y given specific x and v value
        psv2=hyperplane(hyp_x_max,self.w,self.b,1)
        #ploting the associate coordinate of psv2 and psv1 to visualize the hyperplane where v is one ,remeber hyper equation is for y such that v is one
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],"k")
        ##doing the same thing and process for a value of v=-1:
        nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],"k")
        ###doing the same thing and process for a value of v=0:
        db1=hyperplane(hyp_x_min,self.w,self.b,0)
        db2=hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],"y--")
        
        #show the result
        plt.show()
        
        
        
        
data_dict={-1:np.array([[1,7],
                        [2,8],
                        [3,8]]),
            1:np.array([[5,1],[6,-1],[7,3]])}


###trial 1:@19:13 hours\
svm=Support_Vector_Machine()
svm.fit(data=data_dict)
predict_us=[[0,10],[1,3],[3,4],[3,5],[5,5],[6,-5],[5,8]]
for p in predict_us:
    svm.predict(p)
svm.visualize()
############################################-SVM COMPLETED::::

    
    
    
    
    
    