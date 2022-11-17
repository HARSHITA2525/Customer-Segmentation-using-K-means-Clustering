#!/usr/bin/env python
# coding: utf-8

# In[37]:


#importing the library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans #for forming clusters of customer 


# In[2]:


#reding the Dataset
customerData=pd.read_csv("C:\\Users\\Harshita\\Desktop\\customers.csv") #from csv file "customers.csv"


# In[3]:


#printing first 5 rows of teh Data Frame
customerData.head(15)


# In[4]:


#to get the number of rows and columns from the data
customerData.shape


# In[5]:


#Information about the dataset
customerData.info()


# In[6]:


#checking for the mission values in Dataset
customerData.isnull().sum()


# In[7]:


x=customerData.iloc[ : , [3,4] ].values


# In[8]:


x


# In[9]:


#Bar Graph(Gender vs No of Customers)
genders=customerData.Gender.value_counts()
plt.figure(figsize=(7,7))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


# In[10]:


#Bar Graph(Age vs  No of Customer)
df=customerData
age18_25 = df.Age[(df.Age<=25)&(df.Age>=18)]
age26_35 = df.Age[(df.Age<=35)&(df.Age>=26)]
age36_45 = df.Age[(df.Age<=45)&(df.Age>=36)]
age46_55 = df.Age[(df.Age<=55)&(df.Age>=46)]
age55above = df.Age[(df.Age>=56)]


# In[11]:


x=["18-25","26-35","36-45","46-55","Above55"]
y=[len(age18_25.values), len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]


# In[12]:


plt.figure(figsize=(15,6))
plt.title=("Number of customers and ages")
plt.xlabel=("Ages")
plt.ylabel=("Number of customers")
sns.barplot(x=x,y=y)
plt.show()


# In[13]:


#Bar graph(Spending Score vs No of Customers)
ss1_20= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=1) &(df["Spending Score (1-100)"]<=20)]
ss21_40= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=21) &(df["Spending Score (1-100)"]<=40)] 
ss41_60= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=41) &(df["Spending Score (1-100)"]<=60)]
ss61_80= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=61) &(df["Spending Score (1-100)"]<=80)]
ss81_100= df["Spending Score (1-100)"][(df["Spending Score (1-100)"]>=81) &(df["Spending Score (1-100)"]<=100)]


# In[14]:


x=["1-20","21-40","41-60","61-80","81-100"]
y=[len(ss1_20.values), len(ss21_40.values),len(ss41_60.values),len(ss61_80.values),len(ss81_100.values)]


# In[15]:


sns.barplot(x=x , y=y)
plt.figure(figsize=(10,20))
plt.title=("Spending scores of the customers")
plt.xlabel=("Spending Scores")
plt.ylabel=("score of customers")
plt.show()


# In[16]:


#Bar Graph (Annual Income vs No of Customers)
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=0)&(df["Annual Income (k$)"]<=30)] 
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=31)&(df["Annual Income (k$)"]<=60)] 
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=61)&(df["Annual Income (k$)"]<=90)] 
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=91)&(df["Annual Income (k$)"]<=120)] 
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"]>=121)&(df["Annual Income (k$)"]<=150)]


# In[17]:


x=["0-30","31-60", "61-90","91-120","121-150"]
y=[len(ai0_30.values), len(ai31_60.values), len(ai61_90.values),len(ai91_120.values), len(ai121_150.values)]


# In[20]:


plt.figure(figsize=(15,6))
sns.barplot(x=x,y=y,)
plt.title=("Annual Income of customers")
plt.xlabel=("Annual Income in kDollar ")
plt.ylabel=("Number of customers")
plt.show()


# In[25]:


x=customerData.iloc[ : , [3,4] ].values


# In[26]:


x


# In[31]:


#choosing number of clusters
L = []
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',random_state=40 )
    km.fit(x)
    L.append(km.inertia_)


# In[34]:


import matplotlib.pyplot as plt
#plotting the elbow graph.....
sns.set()
plt.plot(range(1,11),L)
plt.title=('Elbow Graph')
plt.xlabel=('Number of clusters')
plt.ylabel=('Sum of Square')
plt.show()


# In[35]:


#Optimum number of cluster will be 5
#Training the Model(K Means Clustering Model)
km=KMeans(n_clusters=5,init='k-means++' ,random_state=0)

y=km.fit_predict(x)
y


# In[41]:


#Visualizing all the clusters

plt.figure(figsize=(8,8))

plt.scatter(x[y==0,0], x[y==0,1], s=50, c='red', label='1')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='yellow', label='2')
plt.scatter(x[y==2,0], x[y==2,1], s=50, c='green', label='3')
plt.scatter(x[y==3,0], x[y==3,1], s=50, c='black', label='4')
plt.scatter(x[y==4,0], x[y==4,1], s=50, c='blue', label='5')

plt.scatter(km.cluster_centers_[ : ,0],km.cluster_centers_[ : ,1], s=100, c='cyan', label='Centroids'    ) 

plt.title=('Customer Groups')
plt.xlabel=("Annual Income")
plt.ylabel=('Spending Score')
plt.show()


# In[ ]:




