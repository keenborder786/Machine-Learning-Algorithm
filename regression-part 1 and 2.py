# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:06:51 2018

@author: MMOHTASHIM
"""
import pandas as pd
import quandl
import math
df=quandl.get("WIKI/GOOGL",auth_token="fwwt3dyY_pF8LyZqpNsa")
df=df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]

df["HL_pct"]=(df["Adj. High"]-df["Adj. Close"])/(df["Adj. Close"])*100
df["PCT_change"]=(df["Adj. Close"]-df["Adj. Open"])/(df["Adj. Open"])*100
df=df[["Adj. Close","HL_pct","PCT_change","Adj. Volume"]]

forecast_col="Adj. Close"
df.fillna(-99999,inplace=True)
print(df)

forecast_out=int(math.ceil(0.01*len(df)))
print(len(df))
df["label"]=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df)