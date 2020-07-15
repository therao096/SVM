# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:16:23 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder


forest=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\SUPPORT VECTOR MACHINES\\forestfires.csv")
forest.columns
forest.shape
forest.describe()
sns.pairplot(forest)
le=LabelEncoder()
forest['month']=le.fit_transform(forest.month)
forest['day']=le.fit_transform(forest.day)
train,test= train_test_split(forest,test_size=0.3, random_state=0)
train_x=train.iloc[:,:30]
train_y=train.iloc[:,30]

test_x=test.iloc[:,:30]
test_y= test.iloc[:,30]


model_linear=SVC(kernel="linear")
model_linear.fit(train_x,train_y)
pred_test_linear=model_linear.predict(test_x)

np.mean(pred_test_linear==test_y)
####accuracy is 97.79

####using "poly kernel"

model_poly= SVC(kernel="poly")
model_poly.fit(train_x,train_y)
pred_poly=model_poly.predict(test_x)
np.mean(pred_poly==test_y)
###accuracy is 75.64%