
import pandas as pd
import quandl
import math,datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import cross_validate,train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")
    
df=quandl.get("WIKI/GOOGL",auth_token="fwwt3dyY_pF8LyZqpNsa")
df=df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]
df["HL_pct"]=(df["Adj. High"]-df["Adj. Close"])/(df["Adj. Close"])*100
df["PCT_change"]=(df["Adj. Close"]-df["Adj. Open"])/(df["Adj. Open"])*100
df=df[["Adj. Close","HL_pct","PCT_change","Adj. Volume"]]
df.to_csv("Project try.csv")
print(df)
forecast_col="Adj. Close"
df.fillna(-99999,inplace=True)


forecast_out=int(math.ceil(0.1*len(df)))
print(len(df))
df["label"]=df[forecast_col].shift(-forecast_out)


X=np.array(df.drop(["label"],1))
X=preprocessing.scale(X)
X_lately=X[-forecast_out:]
X=X[:-forecast_out]


df.dropna(inplace=True)
y=np.array(df["label"])
    
print(df)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


clf=LinearRegression(n_jobs=-1)

clf.fit(X_train,y_train)
with open("linearregression.pickle","wb") as f:
    pickle.dump(clf,f)
pickle_in=open("linearregression.pickle","rb")
clf=pickle.load(pickle_in)

accuracy=clf.score(X_test,y_test)


print(accuracy)

forecast_set=clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)

df["Forecast"]=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
print(last_unix)
one_day=86400
next_unix=last_unix+one_day
print(next_unix)


for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
    print(next_date)
print(df)
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
############################################################################
##Linear Regression Model from scratch:
#from statistics import mean
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use("fivethirtyeight")
#xs=np.array([1,2,3,4,5,6],dtype=np.float64)
#ys=np.array([5,4,6,5,6,7],dtype=np.float64)
#
#def best_fit_slope_and_intercept(xs,ys):
#    m=((mean(xs)*mean(ys)) - (mean(xs*ys)))/((mean(xs)**2)-mean(xs**2))
#    b=mean(ys)-m*mean(xs)
#    return m,b
#m,b=best_fit_slope_and_intercept(xs,ys)
#print(m,b)
#
#regression_line=[(m*x)+b for x in xs]
#predict_x=8
#predict_y=(m*predict_x)+b
#print(regression_line)
#plt.scatter(xs,ys)
#plt.scatter(predict_x,predict_y,color="g")
#plt.plot(regression_line)
    years_temps=[]
    stds=[]
    means=[]
    city_year_temp=[]
    for year in years:
        for city in multi_cities:
            city_year_temp.append(climate.get_yearly_temp(city,year))
        l=len(city_year_temp[0])
        for i in range(l):
            years_temps=[]
            for X in city_year_temp:
                years_temps.append(X[i])
            mean=pylab.mean(years_temps)
            means.append(mean) 
        std=pylab.std(means)
        stds.append(std)
    return pylab.array(stds)
            
    