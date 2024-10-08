# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:33:37 2024

@author: ADMIN
"""

3.	Analyze the information given in the following ‘Insurance Policy dataset’ 
create clusters of persons falling in the same type. Refer to Insurance Dataset.csv

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df=pd.read_csv("D:/7-Clustring/Insurance Dataset.csv")
df.head()
#Now we will check the shape of dataset 
df.shape
#: (100, 5)

#Now we will check the null values of the data
df.isna().sum()
#There are no null values in the data

df.columns
#now we will check the duplicate values
duplicate=df.duplicated()
duplicate
sum(duplicate)
#There are no dulicate values 

df.describe()
#Now we will go for Univarite Analysis
sns.histplot(df["Income"])
sns.histplot(df["Age"])

#Now for the bivariate Analysis
plt.scatter(df["Age"],df["Income"])
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

########################################################
#clustering on all the feature
features=[
    'Premiums Paid',
    'Age',
    'Days to Renew',
    'Claims made',
    'Income'
    ]
#Now we will go for scaling
scaler=MinMaxScaler()
df_scaled=scaler.fit_transform(df[features])
df_scaled

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
#from elnbow curve we will take number of clusters as 3

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df_scaled)
df["cluster"]=y_predicted
km.cluster_centers_
df.head(50)


##########################################
#Now clustering only on two features

#Now we will go Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df=df_norm=norm_func(df.iloc[:,1:])
df.head()

#Determine number of clusters
TWSS=[]
k=list(range(1,9))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df)
    
    TWSS.append(kmeans.inertia_)
    
    
TWSS
plt.plot(k,TWSS,'ro-');
plt.ylabel("No_of_clusters");
plt.xlabel("Total_within_SS")

#from elbow curve we came to known the no_of clusters that is three

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[["Age","Income"]])
y_predicted
df["cluster"]=y_predicted
km.cluster_centers_
y_predicted.shape

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]



plt.scatter(df1.Age,df1["Income"],color="green")
plt.scatter(df2.Age,df2["Income"],color="red")
plt.scatter(df3.Age,df3["Income"],color="black")


plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()


############################################

#now for age and claims made
y_predicted=km.fit_predict(df[["Age","Claims made"]])
y_predicted
df["cluster"]=y_predicted
km.cluster_centers_
y_predicted.shape

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1["Claims made"],color="green")
plt.scatter(df2.Age,df2["Claims made"],color="red")
plt.scatter(df3.Age,df3["Claims made"],color="black")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()



