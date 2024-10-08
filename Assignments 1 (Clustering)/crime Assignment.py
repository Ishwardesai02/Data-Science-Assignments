# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:34:53 2024

@author: ADMIN
"""


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("D:/7-Clustring/crime_data.csv")

#Understand the data 
df.head()
#understand the shape off data 
df.shape
#(50, 5)

#the first column of data set in unnamed so we have to give name
df.columns
col="State"
df.rename(columns={'Unnamed: 0':col}, inplace=True)
df.head()
#now we can see column one that is State has now got the name


df.dtypes
#now we will convert datatype of murder to int
df["Murder"]=df["Murder"].astype(int)
df.dtypes #here we can see murder is converted into the int

#plt.scatter(df.State,df["Murder"],df["Assault"],df["UrbanPop"],df["Rape"])
df.describe()
####################################################
features=[   
    'Murder',
    'Assault',
    'UrbanPop',
    'Rape'
    ]
#Now we will go for scaling
scaler=MinMaxScaler()
df_scaled=scaler.fit_transform(df[features])

#now using elsbow curve we will determine the number of culsters for df_scaled 
TWSS=[]
k=list(range(1,9))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_scaled)
    
    TWSS.append(kmeans.inertia_)
    
TWSS
plt.plot(k,TWSS,'ro-');
plt.ylabel("No_of_clusters");
plt.xlabel("Total_within_SS")

#from the elbow curve we can see the no of clusters is 3 or 4

#if we consider 3 cluster 
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df_scaled)
df["cluster"]=y_predicted
km.cluster_centers_
df.head(20)

#########################################################
#Now we will perform clustering onnly on two features

#we will cluster on Murder and Assualt
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Murder","Assault"]])
y_predicted
df['clusters']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.clusters==0]
df2=df[df.clusters==1]
df3=df[df.clusters==2]

plt.scatter(df1.Murder,df1["Assault"],color='green')
plt.scatter(df2.Murder,df2["Assault"],color="red")
plt.scatter(df3.Murder,df3["Assault"],color="blue")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# =now data is not properly scaled we have toscale the data

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df=norm_func(df.iloc[:,1:])
df.head()
'''
scaler=MinMaxScaler()


scaler.fit(df[['Murder']])
df['Murder']=scaler.transform(df[['Murder']])

scaler.fit(df[['Assault']])
df['Assault']=scaler.transform(df[['Assault']])

scaler.fit(df[['Rape']])
df['Rape']=scaler.transform(df[['Rape']])

plt.scatter(df.State,df['Murder'])
plt.scatter(df.State,df['Assault'])
plt.scatter(df.State,df['Rape'])
df.head()
'''


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Murder","Assault"]])
y_predicted
df['clusters']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.clusters==0]
df2=df[df.clusters==1]
df3=df[df.clusters==2]

plt.scatter(df1.Murder,df1["Assault"],color='green')
plt.scatter(df2.Murder,df2["Assault"],color="red")
plt.scatter(df3.Murder,df3["Assault"],color="blue")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()

#Now we are done for Murder vs Assault now we will go for Rape vs Murder
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["UrbanPop","Rape"]])
y_predicted
df['clusters']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.clusters==0]
df2=df[df.clusters==1]
df3=df[df.clusters==2]

plt.scatter(df1.UrbanPop,df1["Rape"],color='green')
plt.scatter(df2.UrbanPop,df2["Rape"],color="red")
plt.scatter(df3.UrbanPop,df3["Rape"],color="blue")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('UrbanPop')
plt.ylabel("Rape")
plt.legend()


#Now we are done for Murder vs Rape now we will go for Rape vs Assault
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Assault","Rape"]])
y_predicted
df["clusters"]=y_predicted
df.head()
km.cluster_centers_


df1=df[df.clusters==0]
df2=df[df.clusters==1]
df3=df[df.clusters==2]

plt.scatter(df1.Assault,df1["Rape"],color='green')
plt.scatter(df2.Assault,df2["Rape"],color="red")
plt.scatter(df3.Assault,df3["Rape"],color="blue")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
