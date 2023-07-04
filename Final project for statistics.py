#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats
import statsmodels.api as sm


# In[2]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
df = pd.read_csv(boston_url)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# ## Become familiar with data

# The following describes the dataset variables:
# 
# ·      CRIM - per capita crime rate by town
# 
# ·      ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# ·      INDUS - proportion of non-retail business acres per town.
# 
# ·      CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 
# ·      NOX - nitric oxides concentration (parts per 10 million)
# 
# ·      RM - average number of rooms per dwelling
# 
# ·      AGE - proportion of owner-occupied units built prior to 1940
# 
# ·      DIS - weighted distances to five Boston employment centres
# 
# ·      RAD - index of accessibility to radial highways
# 
# ·      TAX - full-value property-tax rate per $10,000
# 
# ·      PTRATIO - pupil-teacher ratio by town
# 
# ·      LSTAT - % lower status of the population
# 
# ·      MEDV - Median value of owner-occupied homes in $1000's

# ## Task 4 : Generate Descriptive statistics and Visualizations

# ### Q1:For the "Median value of owner-occupied homes" provide a boxplot

# In[12]:


a=sns.boxplot(x='MEDV',data=df)
a.set_title('Median Values of owner-occupied homes')


# we can find that there are outliers in the median values of owner occupied homes

# ### Q2: Provide a  bar plot for the Charles river variable

# In[17]:


ax=sns.barplot(y='CHAS',data=df)
ax.set_title("Charles river variable")


# ### Q3:Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)

# In[22]:


df.loc[(df['AGE'] <= 35),'Age_group'] = '35 years and younger'
df.loc[(df['AGE'] > 35)& (df['AGE'] < 70),'Age_group'] = 'between 35 and 70 years'
df.loc[(df['AGE'] >= 70),'Age_group'] = '70 years and older'

ax=sns.boxplot(x='MEDV',y='Age_group',data=df)
ax.set_title('Median values of owner-occupied home by age group')


# ### Q4:Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?

# In[25]:


ax=sns.scatterplot(x='NOX',y='INDUS',data=df)
ax.set_title("Relationship between NOX and INDUS")


# Values in the bottom-left section of the scatter plot indicates a strong relation between low Nitric oxide concentration and low proportion of non-retail business acres per town.
# 
# Generally, a higher proprtion of non-retail business acres per town produces a higher concentration of Nitric oxide.

# ### Q5:Create a histogram for the pupil to teacher ratio variable

# In[26]:


ax=sns.histplot(x='PTRATIO',data=df)
ax.set_title('Distribution of pupil to teacher ratio variable')


# ## Task 5: Use the appropriate tests to answer the questions provided.

# ### Q1:Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

# H0:There is no significant difference in median value of houses bounded by 
#     the Charles river
# H1:There is a significant difference in median value of houses bounded by 
#     the Charles river

# In[29]:


df.loc[(df['CHAS']==0),'CHAS_T']='far'
df.loc[(df['CHAS']==1),'CHAS_T']='near'

scipy.stats.ttest_ind(df[df['CHAS_T'] == 'far']['MEDV'],
                   df[df['CHAS_T'] == 'near']['MEDV'], equal_var = True)


# Here we can see that the pvalue is 0.00007 and the statistic value which we are 
# calculated is -3.99 which is statistic values > p values, so we reject the H0(null hypothesis)
# 
# #### Inference:
#     There is a significant differnece in median values of house bounded by charles river

# ### Q2:Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

# Null Hypotesis(H0): There is no statistical
#     difference in Median values of houses (MEDV) for each proportion of 
#     owner occpied units built prior to 1940
# 
# Alternative Hypothesis(H1): There is statistical
#     difference in Median values of houses (MEDV)
#     for each proportion of owner occpied units built prior to 1940

# In[34]:


from statsmodels.formula.api import ols
lm=ols('MEDV ~ AGE',data=df).fit()
table=sm.stats.anova_lm(lm)
table


# Given that the pvalue is less than the statistic value so we reject the null hypotheis H0
# 
# #### Inference:
#     There is statistical difference in Median values of houses (MEDV) for each proportion of owner occpied units built prior to 1940
#     

# ### Q3:Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

# Null Hypothesis(H0): Nitric Oxide concentration is not correlated with the proportion of non-retail business acres per town
# 
# Alternative Hypothesis(H1): Nitric Oxide concentration is correlated with the proportion of non-retail business acres per town

# In[36]:


scipy.stats.pearsonr(df['NOX'],df['INDUS'])


# Here the p-value is less than the statistic values so we reject the null hypothesis
# 
# #### Inference:
# 
#     Nitric Oxide concentration is correlated with the proportion of non-retail business acres per town

# ### Q4:What is the impact of an additional weighted distance  to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

# In[37]:


x = df['DIS']
y = df['MEDV']

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predisction = model.predict(x)

model.summary()


# The coef DIS of 1.0916 indicates that an additional weighted distance to the 5 empolyment centers in boston increases of 1.0916 the median value of owner occupied homes

# In[ ]:




