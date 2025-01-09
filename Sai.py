#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Module is a collection of functions.


# In[16]:


def  mean_value(*n):
    sum=0
    counter=0
    for x in n:
        counter+=1
        sum+=x
    mean=sum/counter
    return mean


# In[18]:


mean_value(10,39,20)


# In[12]:


def product(*n):
    result=1
    for i in range(len(n)):
        result *= n[i]
    return result


# In[14]:


product(20,30,40)


# In[ ]:





# In[ ]:




