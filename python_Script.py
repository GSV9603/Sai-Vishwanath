#!/usr/bin/env python
# coding: utf-8

# In[1]:


def mean_value(given_list):
    total=sum(given_list)
    average_value=total/len(given_list)
    return average_value
L=[10,11,12,12,14]
mean_value(L)


# In[14]:


def mode(s):
     s=set(L)
     d={}
     for x in s:
         d[x]=L.count(x)
     m=max(d.values())
     for k in d.keys():
         if d[k]==m:
             return k


# In[18]:


L=["S","A","I","S","A","I","A","A"]
mode(L)


# In[20]:


L=[2,4,2,1,3.2,2,1,3,5,5,]
mode(L)


# In[ ]:




