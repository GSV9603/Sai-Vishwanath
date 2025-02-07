#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 1. Linearity: The relationship between the predictors and the response is linear.
# 2. Independence: Observations are independent of each other.
# 3. Homoscedasticity: The residuals (Y - Y hat)) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other.
# Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions
# The general formula for multiple linear regression is:

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP", "WT","MPG"])
cars.head()


# Description of columns
# 1. MPG : Milege of the car (Mile per Gallon)(This is Y-column to be predicted)
# 2. HP : Horse Power of the car(X1 column)
# 3. VOL : Volume of the car(Miles per Hour)(X3 column)
# 4. SP : Top speed of the car(Miles of the car(Miles Per Hour)(X3 column)
# 5. WT : Weight of the car(Pounds)(X4 column)

# In[6]:


cars.info()


# In[7]:


cars.isna().sum()


# Observations
# 1. There are no missing values
# 2. There are 81 observations (81 different cars data)
# 3. The data type of the column are also relevant and valid

# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[13]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# Observations from boxplot and histograms
# 1. There are some extreme values(outliers) observed in towards the right tail of SP and HP distributions.
# 2. In VOL and WT columns, a few outliers are observed in both tails of their distributions.
# 3. The extreme values of cars data may have come from the specially designed of cars.
# 4. As this multi-dimensional data,the outliers with respect to spatial dimensions may have to be considered while building the resgression model.

# In[15]:


cars[cars.duplicated()]


# In[16]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[17]:


cars.corr()


# ### Observations from correlation plots and Coeffcients
# - Between x and y, all the x variables are shoeing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for buliding a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3, and x4),some very high correlation strengths are observed between SP vs HP,VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem

# ### Preparing a preliminary model considering all X columns

# In[20]:


import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[21]:


model1.summary()


# ### Observations from model summary
# 
# - The R-squared and adjusted R-suared values are good and about 75% of varuability in Y is explained by X colums
# - The probabaility value with respect to F_stastic is close to zero indicating that all are some of x columns are significant
# - The p-values for VOL and WT are higer than 5% indicating some intercation issue among themselves which need to be further explored

# ### Performance metrics for model1

# In[24]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[25]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[26]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE :", mean_squared_error(df1["actual_y1"], df1["pred_y1"]))


# In[27]:


pred_y1=model1.predict(cars.iloc[:,0:4])
df1["pred_y1"]=pred_y1
df1.head()


# In[48]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# Checking for multicllinearity among X-columns using VIF method

# In[51]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)

# Storing vif values in a data frame
d1 = {'Variables' : ['Hp' ,'WT' , 'VOL' ,'SP' ], 'VIF' : [vif_hp,vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)
Vif_frame


# Observations:
# * The ideal range of VIF values shall be between 0 to 10. However slightly high values can be tolerated.
# * As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity proble.
# * Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity.
# * It is decided to drop WT and retain VOL column in further models

# In[54]:


cars1=cars.drop("WT", axis=1)
cars1.head()


# In[56]:


model2=smf.ols('MPG~VOL+SP+HP', data=cars1).fit()


# In[58]:


model2.summary()


# In[60]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### Observations from model2 summary()
# * The adjusted R-suared value improved slightly to 0.76
# * All the p-values for model parametersa re less than 5% hence they are significant
# * Therefore the HP, VOL, SP columns are finalized as the significant predictor for the MPG response vriabl
# * There is no improvement in MSE value

# ### Performance metrics for model2

# In[62]:


df2=pd.DataFrame()
df2["actual_y2"]=cars["MPG"]
df2.head()


# In[ ]:


pred_y2=model2.predict(cars.iloc[:,0:4])
df2["]

