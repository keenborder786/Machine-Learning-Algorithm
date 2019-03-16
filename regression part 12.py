# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 01:49:11 2019

@author: MMOHTASHIM
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 14:15:33 2019

@author: MMOHTASHIM
"""

#Linear Regression Model from scratch:
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use("fivethirtyeight")
#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
#This fuction is to test the accuracy of our assumptions
def create_dataset(hm,variance,step=2,correlation=False):
    val=1
    ys=[]
    for y in range(hm):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation=="pos":
            val+=step
        elif correlation and correlation=="negative":
            val-=step
    xs=[i for i in range(len(ys))]
    
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)
    
    


def best_fit_slope_and_intercept(xs,ys):
    m=((mean(xs)*mean(ys)) - (mean(xs*ys)))/((mean(xs)**2)-mean(xs**2))
    b=mean(ys)-m*mean(xs)
    return m,b

def squared_error(ys_orgin,ys_line):
    return sum((ys_line-ys_orgin)**2)
def coefficent_of_determination(ys_orgin,ys_line):
    y_mean_line=[mean(ys_orgin) for y in ys_orgin]
    square_error_regr=squared_error(ys_orgin,ys_line)
    square_error_regr_y_mean=squared_error(ys_orgin,y_mean_line)
    return 1-(square_error_regr)/(square_error_regr_y_mean)
xs,ys=create_dataset(40,5,2,correlation="negative")





m,b=best_fit_slope_and_intercept(xs,ys)
print(m,b)

regression_line=[(m*x)+b for x in xs]
r_square=coefficent_of_determination(ys,regression_line)
print(r_square)
x_predict=8
y_predict=[(m*x_predict)+b]
plt.scatter(xs,ys)
plt.scatter(x_predict,y_predict,s=10,color="green")
plt.scatter(xs,ys)
plt.plot(regression_line)
plt.show()