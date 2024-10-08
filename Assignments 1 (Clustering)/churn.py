# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:49:08 2024

@author: ADMIN
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("D:/7-Clustring/Telco_customer_churn.csv")
df.head()
df.shape
df.describe()

df.dtypes

#Now check the null values
df.isna().sum()
df=df.dropna()
df.shape
#We have dropped the null values using dropna() function

#Now we will check that is there any duplicate value
duplicate=df.duplicated()
sum(duplicate)
#there are no duplicate values as it return 0 sum

#Multivariant analysis
sns.pairplot(df)

#Now scale the data using StandardSacler function

rename_dict = {
    'Tenure in Months': 'Tenure_in_Months',
    'Avg Monthly Long Distance Charges': 'Avg_Monthly_Long_Distance_Charges',
    'Avg Monthly GB Download':'Avg_Monthly_GB_Download',
    'Monthly Charge':'Monthly_Charge',
    'Total Charges': 'Total_Charges',
    'Total Extra Data Charges':'Total_Extra_Data_Charges',
    'Total Long Distance Charges':'Total_Long_Distance_Charges',
    'Total Revenue':'Total_Revenue'
    
}
# Rename columns
df.rename(columns=rename_dict, inplace=True)

df.columns

# List of features to Standardize
features = [
    'Tenure_in_Months', 
    'Avg_Monthly_Long_Distance_Charges',
    'Avg_Monthly_GB_Download',
    'Monthly_Charge', 
    'Total_Charges', 
    'Total_Extra_Data_Charges',
    'Total_Long_Distance_Charges',
    'Total_Revenue'
]
scaler=StandardScaler()
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
    
#on the basis of elbow curve we will we can use 3 or 4 number of cluster


#Now we will apply k means clustering
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df_scaled)
df["cluster"]=y_predicted
km.cluster_centers_
df.head(20)

############################################################################

#Only for two features
# Initialize the scaler
scaler=MinMaxScaler()

scaler.fit(df[['Total_Revenue']])
df['Total_Revenue']=scaler.transform(df[['Total_Revenue']])

scaler.fit(df[['Total_Charges']])
df['Total_Charges']=scaler.transform(df[['Total_Charges']])

plt.scatter(df.Total_Charges,df['Total_Revenue'])




#Determine number of clusters
TWSS=[]
k=list(range(1,9))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[["Total_Charges","Total_Revenue"]])
    
    TWSS.append(kmeans.inertia_)
    
    
TWSS
plt.plot(k,TWSS,'ro-');
plt.ylabel("No_of_clusters");
plt.xlabel("Total_within_SS")

km=KMeans(n_clusters=3)
df["cluster"]=km.fit_predict(df[["Total_Charges","Total_Revenue"]])
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Total_Charges,df1["Total_Revenue"],color="green")
plt.scatter(df2.Total_Charges,df2["Total_Revenue"],color="red")
plt.scatter(df3.Total_Charges,df3["Total_Revenue"],color="black")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()



