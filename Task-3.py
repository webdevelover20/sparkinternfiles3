#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


ds=pd.read_csv('Iris.csv')
ds


# In[3]:


x=ds.iloc[:,1:5].values


# In[4]:


x


# In[5]:


#scatter plot between first two columns in x
plt.scatter(x[:,0],x[:,1])


# In[6]:


from sklearn.cluster import KMeans
wcss=[]


# In[7]:


for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)  #to calculate wcss, method is inertia_
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# As WCSS doesn't decrease significantly we should consider k as 3 i.e.,number of clusters as 3

# In[8]:


k=KMeans(n_clusters=3,init='k-means++',random_state=0)
ymeans=k.fit_predict(x)
ymeans


# In[9]:


k.cluster_centers_


# In[10]:


plt.scatter(x[ymeans==0,0],x[ymeans==0,1],s=100,color='red',label='verginica')
plt.scatter(x[ymeans==1,0],x[ymeans==1,1],s=100,color='green',label='setosa')
plt.scatter(x[ymeans==2,0],x[ymeans==2,1],s=100,color='cyan',label='versicolour')
plt.scatter(k.cluster_centers_[:,0],k.cluster_centers_[:,1],s=300,color='yellow',label='centroids')
plt.title('clusters of customers')
plt.xlabel('sepal length in cm')
plt.ylabel('Sepal width in cm')
plt.legend()
plt.show()


# In[11]:


plt.scatter(x[ymeans==0,2],x[ymeans==0,3],s=100,color='red',label='verginica')
plt.scatter(x[ymeans==1,2],x[ymeans==1,3],s=100,color='green',label='setosa')
plt.scatter(x[ymeans==2,2],x[ymeans==2,3],s=100,color='cyan',label='versicolour')
plt.scatter(k.cluster_centers_[:,2],k.cluster_centers_[:,3],s=300,color='yellow',label='centroids')
plt.title('clusters of customers')
plt.xlabel('petal length in cm')
plt.ylabel('petal width in cm')
plt.legend()
plt.show()

