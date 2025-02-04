#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 
# 1. Linearity: The relationship between the predictors and the response is linear.
# 
# 2. Independence: Observations are independent of each other.
# 
# 3. Homoscedasticity: The residuals (Y - Y_hat)) exhibit constant variance at all levels of the predictor.
# 
# 4. Normal Distribution of Errors: The residuals of the model are normally distributed.
# 
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other.
# 

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels. formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[5]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# Description of columns
# 
# * MPG : Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# * HP : Horse Power of the car (X1 column)
# * VOL : Volume of the car (size) (X2 column)
# * SP : Top speed of the car (Miles per Hour) (X3 column)
# * WT : Weight of the car (Pounds) (X4 Column)

# In[8]:


cars.info()


# In[10]:


cars.isnull().sum()


# In[ ]:




