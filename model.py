# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:56:44 2020

@author: jkalan
"""


#############
## Imports ##
#############
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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




# =============================================================================
# retrieved data and column info from:
# https://archive.ics.uci.edu/ml/datasets/census+income
# =============================================================================

column_names = [ 'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation'
                ,'relationship','race','sex','capital_gain','capital_loss', 'hrs_per_week', 'native_country', 'income']
train_set = pd.read_csv('adult-training.csv', names =column_names)

test_set = pd.read_csv('adult-test.csv',skiprows=1,names = column_names)

##get some initial info
train_set.info()
test_set.info()
##remove whitespaces and deal with ? values
train_set.replace(' ?',np.nan,inplace=True)
test_set.replace(' ?',np.nan,inplace=True)

##descr table will have std dev and mean values for use in extreme_cap fn 
descr_table = train_set.describe()
a= test_set.describe()

##counts/unique values
for name in column_names:
    print(train_set[name].value_counts())
##unique values
for name in column_names:
    print(name, train_set[name].unique())
 
#cols_to_filter =["age","education_num","capital_gain","hrs_per_week","fnlwgt","capital_loss"]

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
train_set = extreme_cap(train_set, 1.5)
##cap extreme values on test set 
test_set = extreme_cap(test_set, 1.5)



#####################
## Pre-processing ##
####################

#convert income to binary output
train_set["income"] = train_set["income"].apply(lambda x: 0 if x==' <=50K' else 1)
test_set["income"] = test_set["income"].apply(lambda x: 0 if x==' <=50K.' else 1)


#split target from features
train_features, train_target = train_set.iloc[:,:-1], train_set.iloc[:,-1]
test_features, test_target = test_set.iloc[:,:-1], test_set.iloc[:,-1]

#one-hot encoding for categorical variables
col_cat =["workclass","education","marital_status","occupation","relationship","race","sex","native_country"] 

for name in col_cat:
    name_df = pd.DataFrame( train_features[name])
    dum_df  = pd.get_dummies(name_df, prefix=[name+'_'] )
    train_features = train_features.join(dum_df)
    
train_features.drop(col_cat, axis='columns',inplace=True)
 

##one-hot-encoding test features
for name in col_cat:
    name_df = pd.DataFrame(test_features[name])
    dum_df  = pd.get_dummies(name_df, prefix=[name+'_'] )
    test_features = test_features.join(dum_df)
    
test_features.drop(col_cat, axis='columns',inplace=True)

##add column native_country__ Holand-Netherlands so that columns are same in test and train
test_features["native_country__ Holand-Netherlands"]=0
#sorting test and train features so order is identical
test_features = test_features.reindex(sorted(test_features.columns), axis=1)
train_features = train_features.reindex(sorted(train_features.columns), axis=1)


######################
## Model 1: XGBoost ##
######################

model = xgb.XGBClassifier( objective= 'binary:logistic'
                         , colsample_bytree= 0.3
                         , learning_Rate =.1
                         , max_depth=10
                         , verbosity =0
                         )
model.fit(train_features, train_target)
preds =model.predict(test_features)


#######################
## XGB Performance   ##
#######################

xgb_roc = roc_auc_score(y_true=test_target, y_score=preds)
print('AUC:', round(xgb_roc*100,2))



############################
## Model 2 - RandomForest ##
############################

model_rf = RandomForestClassifier(n_estimators= 100 ##num trees
                                 , max_depth=10     ##num levels in forest
                                 , random_state=0   
                                 , min_samples_splitint =2  #min num samples to split on
                                 , min_samples_leaf = 2    #min samples at each node
                                 )

model_rf.fit(train_features, train_target)

rf_preds = model_rf.predict(test_features)

roc = roc_auc_score(y_true=test_target, y_score=rf_preds)
print('AUC:', round(roc*100,2))
precision_score(y_true=test_target, y_pred= rf_preds, average='weighted')

#############################
##  Hyperparameter Tuning ##
############################


grid = { 'n_estimators': list(range(75,300, 25))
       , 'max_depth': list(range(5,50, 5))
       , 'min_samples_split': [2,4,6]
       , 'min_samples_leaf': [1,2,3,4]
       , 'bootstrap': [True,False]}

rf_tuning = RandomizedSearchCV(  estimator=model_rf
                               , param_distributions= grid
                               , n_iter = 10
                               , scoring='roc_auc'
                               , cv = 5
                               , verbose=2
                               , random_state=0
                               , return_train_score=True)

## this will take a few minutes
rf_tuning.fit(train_features, train_target)

best_params_rf= rf_tuning.best_params_
best_model = rf_tuning.best_estimator_
best_model.fit(train_features,train_target)
best_preds = best_model.predict(test_features)

best_roc = roc_auc_score(y_true=test_target, y_score=best_preds)
print('Tuned AUC:', round(best_roc*100,2))
print('Initial AUC:', round(roc*100,2))