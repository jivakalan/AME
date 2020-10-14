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

##missing values --appear to be none but there are '?' values 
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
 
#convert income to binary output
train_set["income"] = train_set["income"].apply(lambda x: 0 if x==' <=50K' else 1)
test_set["income"] = test_set["income"].apply(lambda x: 0 if x==' <=50K.' else 1)

    
sum(train_set['income']) 
sum(test_set['income']) 
# TRAIN
#>50k -- 7841/32561  24%
# <50k --            76%
train_set.duplicated(subset=None, keep='first').value_counts()
# 24 duplicates

# TEST
#>50k -- 3846/16281  23.6%
# <50k --            76.4%
test_set.duplicated(subset=None, keep='first').value_counts()
# 5 duplicates

#############################
##visual audit of columns###
############################

##countplots
fig,ax=plt.subplots(4,2,figsize=(20,30))
a= sns.countplot(x=train_set["workclass"],ax=ax[0,0])
a.set_xticklabels(a.get_xticklabels(), rotation=45, horizontalalignment='right')
b= sns.countplot(x=train_set["education"], ax=ax[0,1])
b.set_xticklabels(b.get_xticklabels(), rotation=45, horizontalalignment='right')
c= sns.countplot(x=train_set["marital_status"], ax=ax[1,0])
c.set_xticklabels(c.get_xticklabels(), rotation=45, horizontalalignment='right')
sns.countplot(x=train_set["relationship"],ax=ax[1,1])
sns.countplot(x=train_set["sex"], ax=ax[2,0])
d= sns.countplot(x=train_set["occupation"], ax=ax[3,0])
d.set_xticklabels(d.get_xticklabels(), rotation=45, horizontalalignment='right')
e= sns.countplot(x=train_set["race"], ax=ax[3,1])
e.set_xticklabels(e.get_xticklabels(), rotation=45, horizontalalignment='right')
fig.show()

# Which class in the above variables have the highest representation
# workclass: private
# education: bachelors, HS grad, some-college
# marital status: never-married; married-civ-spouse, divorced
# relationship: not-in-famil;husband;own-child
# sex: male
# occupation: fairly even, minimal armed forces (only 9)
# race : white
############################

##histograms
fig,ax=plt.subplots(2,3,figsize=(15,5))
train_set.hist("age",ax=ax[0,0])
train_set.hist("education_num", ax=ax[0,1])
train_set.hist("capital_gain", ax=ax[0,2])
train_set.hist("hrs_per_week", ax=ax[1,0])
train_set.hist("fnlwgt", ax=ax[1,1])
train_set.hist("capital_loss", ax=ax[1,2])

# age: skews young
# fnlwgt : not sure what this represents: supposedly it is some kind of estimate of
#         population total but when a straight sum of this column is >6Billion


#visualize extreme values
fig,ax=plt.subplots(2,3,figsize=(15,5))
train_set.boxplot("age", ax=ax[0,0])
train_set.boxplot("education_num", ax=ax[0,1])
train_set.boxplot("capital_gain", ax=ax[0,2])
train_set.boxplot("fnlwgt", ax=ax[1,1])
train_set.boxplot("capital_loss", ax=ax[1,2])
train_set.boxplot("hrs_per_week",ax=ax[1,0])

