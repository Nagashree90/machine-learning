#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[23]:





# In[24]:





# In[ ]:





# In[41]:


import pandas as pd
dataframe=pd.read_csv('diabetes.csv')


# In[2]:


import pandas as pd
dataframe=pd.read_csv('diabetes.csv')


# In[3]:


dataframe.head()


# In[4]:


array = dataframe.values


# In[26]:


array


# In[6]:


array


# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[8]:


import seaborn as sns
corr = dataframe.corr()


# In[9]:


corr


# In[10]:


y = dataframe.Outcome


# In[11]:


y


# In[14]:


feature=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x = dataframe[feature]


# In[15]:


x.head()


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[17]:


clf_d=DecisionTreeClassifier()


# In[19]:


clf_d=clf_d.fit(x_train,y_train)


# In[20]:


y_pred=clf_d.predict(x_test)


# In[21]:


y_pred


# In[28]:


y_test
metrics.accuracy_score(y_test,y_pred)


# In[29]:


y_test


# In[31]:


from sklearn.metrics import confusion_matrix


# In[33]:


print(confusion_matrix(y_test,y_pred))


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[35]:


rf.fit(x_train,y_train)


# In[36]:


rf.score(x_train,y_train)


# In[38]:


rf.score(x_test,y_test)


# In[39]:


y_pred= rf.predict(x_test)


# In[40]:


y_pred


# In[ ]:





# In[ ]:





# In[5]:





# In[6]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:




