# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:56:44 2020

@author: jkalan
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import statistics as st
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
descr_table = train_set.describe()
##counts
for name in column_names:
    print(train_set[name].value_counts())
##unique values
for name in column_names:
    print(name, train_set[name].unique())
 
    
#############################
##visual audit of columns###
############################

##bar charts
    ###fix labels!!!!!! 
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

cols_to_filter =["age","education_num","capital_gain","hrs_per_week","fnlwgt","capital_loss"]



#function to cap extreme values 
def extreme_cap(df, num_std_dev):
    df_capped = df
    for col in descr_table:
       # v = st.stdev(df[col])*num_std_dev
        v = descr_table[col]["std"]*num_std_dev
        hi = descr_table[col]["mean"]+v
        lo = descr_table[col]["mean"]-v
        df_capped[col]=df_capped[col].apply(lambda x: hi if x > hi else x)
        df_capped[col]=df_capped[col].apply(lambda x: lo if x < lo else x)
    return df_capped

#cap values in train set
train_set_capped = extreme_cap(train_set, 1.5)

##cap extreme values on test set 
test_set_capped = extreme_cap(test_set, 1.5)

# =============================================================================
# 
# o	Create a model using these variables (you can use whichever variables you want,
#    or even create you own; for example, you could find the ratio or relationship 
#   between different variables, the binarization of “categorical” variables, etc.) 
#   to determine an income >= $50,000 / year (binary target). 
#
# o	You are free to choose any models as long as there’s more than 2 different types.
#
# o	Choose the model that appears to have the highest performance based on a comparison
#   between reality (the 42nd variable) and the model’s prediction.
#
# o	Apply your model to the test file and measure its real performance on it 
#   same method as above).
# 
# =============================================================================


import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

#####################
## Pre-processing ##
####################

#convert income to binary output
train_set["income"] = train_set["income"].apply(lambda x: 0 if x==' <=50K' else 1)
#one-hot encoding for categorical variables

# creating initial dataframe
bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# generate binary values using get_dummies
dum_df = pd.get_dummies(bridge_df, columns=["Bridge_Types"], prefix=["Type_is"] )
# merge with main df bridge_df on key values
bridge_df = bridge_df.join(dum_df)
bridge_df





#split target from features
train_features, train_target = train_set.iloc[:,:-1], train_set.iloc[:,-1]
test_features, test_target = test_set.iloc[:,:-1], test_set.iloc[:,-1]



############
## Model ##
###########

model = xgb.XGBClassifier( objective='binary: logistic'
                         , colsample_bytree= 0.3
                         , learning_Rate =.1
                         , max_depth=3  
                         , alpha=10  ##l1 regularization
                         )
model.fit(train_features, train_target)
print(model)
preds =model.predict(test_features)


#######################
## Model Performance ##
#######################

roc = roc_auc_score(y_true=test_target, y_score=preds)
print('AUC:', roc)

#######################
## Model Tuning ##
#######################

##grid search to tune hyperparameters

