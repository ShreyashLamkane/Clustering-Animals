#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data=pd.read_csv("zoo.csv")


# In[2]:


data.info()


# In[4]:


import numpy as np
labels =data['class_type']
print(np.unique(labels.values))

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig,ax=plt.subplots()
(labels.value_counts()).plot(ax=ax,kind='bar')


# In[5]:


data.head()


# In[6]:


features=data.values[:,1:-1]
features.shape


# In[7]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

model=AgglomerativeClustering(n_clusters=7,linkage='average',affinity='cosine')


# In[8]:


model.fit(features)


# In[9]:


model.labels_


# In[10]:


print(np.unique(model.labels_))


# In[11]:


labels=labels-1


# In[12]:


from sklearn.metrics import mean_squared_error


# In[13]:


score=mean_squared_error(labels,model.labels_)


# In[14]:


abs_error=np.sqrt(score)
print(abs_error)


# In[ ]:




