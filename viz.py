# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:37:11 2020

@author: jkalan
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# retrieved data and column info from:
# https://archive.ics.uci.edu/ml/datasets/census+income
# =============================================================================

column_names = [ 'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation'
                ,'relationship','race','sex','capital_gain','capital_loss', 'hrs_per_week', 'native_country', 'income']
train_set = pd.read_csv('adult-training.csv', names =column_names)

test_set = pd.read_csv('adult-test.csv',skiprows=1,names = column_names)

# =============================================================================
# 
# Make a quick statistic based and univariate audit of the different columns’ 
# content and produce the results in visual / graphic format.
#
# This audit should describe the variable distribution, the % of missing values
# , the extreme values, and so on.
# 
# =============================================================================

##missing values --appear to be none
train_set.info()
test_set.info()
##remove whitespaces
train_set.replace(' ?',np.nan,inplace=True)
test_set.replace(' ?',np.nan,inplace=True)

descr_table = train_set.describe()
a= test_set.describe()
##counts
for name in column_names:
    print(train_set[name].value_counts())
##unique values
for name in column_names:
    print(name, train_set[name].unique())
 
    
#############################
##visual audit of columns###
############################

##countplots
    ###TO DO - fix labels
fig,ax=plt.subplots(4,2,figsize=(15,15))
sns.countplot(x=train_set["workclass"],ax=ax[0,0])
sns.countplot(x=train_set["education"], ax=ax[0,1])
sns.countplot(x=train_set["marital_status"], ax=ax[1,0])
sns.countplot(x=train_set["relationship"],ax=ax[1,1])
sns.countplot(x=train_set["sex"], ax=ax[2,0])
sns.countplot(x=train_set["occupation"], ax=ax[3,0])
sns.countplot(x=train_set["race"], ax=ax[3,1])
fig.show()

##histograms
fig,ax=plt.subplots(2,3,figsize=(15,5))
train_set.hist("age",ax=ax[0,0])
train_set.hist("education_num", ax=ax[0,1])
train_set.hist("capital_gain", ax=ax[0,2])
train_set.hist("hrs_per_week", ax=ax[1,0])
train_set.hist("fnlwgt", ax=ax[1,1])
train_set.hist("capital_loss", ax=ax[1,2])

#visualize extreme values
fig,ax=plt.subplots(2,3,figsize=(15,5))
train_set.boxplot("age", ax=ax[0,0])
train_set.boxplot("education_num", ax=ax[0,1])
train_set.boxplot("capital_gain", ax=ax[0,2])
train_set.boxplot("fnlwgt", ax=ax[1,1])
train_set.boxplot("capital_loss", ax=ax[1,2])
train_set.boxplot("hrs_per_week",ax=ax[1,0])

