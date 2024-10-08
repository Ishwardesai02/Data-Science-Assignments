# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:25:05 2024

@author: ADMIN
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("D:/7-Clustring/EastWestAir.csv")
df.head()
df.shape

df.isna().sum()
#there are zero null values

df.describe()
#we get five number summary of the dataset


df.columns
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Balance","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12"]])
y_predicted
df['clusters']=y_predicted
df.head(10)
km.cluster_centers_


features = ["Balance", "Bonus_miles", "Bonus_trans", "Flight_miles_12mo", "Flight_trans_12"]
df = df.dropna(subset=features)
# Standardize the features
scaler = StandardScaler()
df = scaler.fit_transform(df[features])


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df)
y_predicted
df['clusters']=y_predicted
km.cluster_centers_


df1=df[df.clusters==0]
df2=df[df.clusters==1]
df3=df[df.clusters==2]

plt.scatter(df1.Balance,df1["Bonus"],color='green')
plt.scatter(df2.Balance,df2["Bonus"],color="red")
plt.scatter(df3.Balance,df3["Bonus"],color="blue")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
