#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewsPaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.show()


# In[8]:


sns.histplot(data1['sunday'], kde = True, stat='density')
plt.show()


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Box plot for sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[10]:


# Scatter plot and Correlation Strength
x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[13]:


model1.summary()


# In[14]:


# plot the scatter plot and overlay the fitted straight line using matplotib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y,  color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
 #predicated response vector
y_hat = b0 + b1*x

# plotting the regression line
plt.plot(x, y_hat, color = "g")

# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[15]:


# predict for 200 and 300 daily circulation
newdata = pd.Series([200,300,1500])


# In[16]:


data_pred = pd.DataFrame(newdata,columns=["daily"])
data_pred


# In[17]:


model1.predict(data_pred)


# In[18]:


# plot the linear regression line unsing  seaborn regression regplot() method
sns.regplot(x="daily", y = "sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[19]:


# predict on all given training data
pred = model1.predict(data1["daily"])
pred


# In[20]:


# add predicated values as a column in data1
data1["Y_hat"] = pred
data1


# In[21]:


# compute the errors values (residuals) and add as another column
data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[22]:


# compute Mean squared Error for the model 
mse = np.mean((data1["daily"]-data1["Y_hat"]) ** 2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[23]:


# Compute Mean Absolute Error (MAE)

mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[24]:


# plot the residuals versus y_hat (to check wheather residuals are independent of error)
plt.scatter(data1["Y_hat"], data1["residuals"])


# In[50]:


import statsmodels.api as sm
sm.qqplot(data1["residuals"],line='45', fit=True)
plt.show()


# In[52]:


sns.histplot(data1["residuals"], kde = True)


# In[54]:


data2 = pd.read_csv("NewspaperData.csv")
data2


# In[ ]:




