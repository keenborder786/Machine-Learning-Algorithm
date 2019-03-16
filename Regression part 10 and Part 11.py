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
import pylab
style.use("fivethirtyeight")
xs=np.array([1,2,3,4,5,6],dtype=np.float64)
ys=np.array([5,4,6,5,6,7],dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m=((mean(xs)*mean(ys)) - (mean(xs*ys)))/((mean(xs)**2)-mean(xs**2))
    b=mean(ys)-m*mean(xs)
    return m,b
m,b=best_fit_slope_and_intercept(xs,ys)
print(m,b)


def squared_error(ys_orgin,ys_line):
    return sum((ys_line-ys_orgin)**2)
def coefficent_of_determination(ys_orgin,ys_line):
    y_mean_line=[mean(ys_orgin) for y in ys_orgin]
    square_error_regr=squared_error(ys_orgin,ys_line)
    square_error_regr_y_mean=squared_error(ys_orgin,y_mean_line)
    return 1-(square_error_regr)/(square_error_regr_y_mean)
regression_line=[(m*x)+b for x in xs]


r_square=coefficent_of_determination(ys,regression_line)
print(r_square)
predict_x=8
predict_y=(m*predict_x)+b
print(regression_line)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color="g")
plt.plot(regression_line)
