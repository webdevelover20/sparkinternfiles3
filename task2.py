#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[29]:


data.plot(kind='scatter',x='Hours',y='Scores')
plt.title('Hours vs percentage')
plt.show()


# In[30]:


data.shape


# In[31]:


data.describe()


# In[32]:


sns.pairplot(data)


# In[33]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[34]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[35]:


from sklearn.linear_model import LinearRegression
k=LinearRegression()
k.fit(X_train,y_train)
print("Data training finished ")


# In[36]:


print(k.coef_)


# In[37]:


print(k.intercept_)


# In[38]:


regression_line=k.coef_*X+k.intercept_
plt.scatter(X,y)
plt.plot(X,regression_line,color='red')
plt.show()


# In[39]:


print(X_test)
y_pred=k.predict(X_test)


# In[40]:


r=pd.DataFrame({'actual':y_test,'predicted':y_pred})
r


# In[47]:


import numpy as np
arr=np.array([[9.25]])


# In[48]:


arr.reshape(-1,1)


# In[50]:


pred_y=k.predict(arr)
print("No of Hours = {}".format(arr))
print("Predicted Score = {}".format(pred_y[0]))


# In[ ]:





# In[ ]:





# In[ ]:




