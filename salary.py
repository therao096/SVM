# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:08:15 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
data_train=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\SUPPORT VECTOR MACHINES\\SalaryData_Train(1).csv")
data_test=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\SUPPORT VECTOR MACHINES\\SalaryData_Test(1).csv")
le=LabelEncoder()
data_train.dtypes
predictors=data_train.iloc[:,:13]
target=data_train.iloc[:,13]

def datatypes(data):
    a=data.dtypes
    g=dict(a)
    le=LabelEncoder()
    for key in g.keys():
        if g[key]==np.object:
            data[key]=le.fit_transform(data[key])
    return data

newpredictors=datatypes(predictors)
traindata=['newpredictors','target']

test_y=data_test.iloc[:,13]

model_linear=SVC(kernel="linear")            
model_linear.fit(newpredictors,target)    
model_predict=model_linear.predict(test_y)
####3np.mean(model_predict==test_y)
pd.crosstab(test_y,model_predict)
