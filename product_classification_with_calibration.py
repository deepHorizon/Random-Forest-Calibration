# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:59:22 2017

@author: Shaurya Rawat
"""

import pandas as pd
X=pd.read_csv("D:\\Kaggle\\Otto group product classification challenge\\train.csv\\train.csv")

#train.head(5)
#
#test=pd.read_csv("D:\\Kaggle\\Otto group product classification challenge\\test.csv\\test.csv")
#
#test.head(5)
#
#train.info()
#test.info()

X['target']=X['target'].map({'Class_1':1,'Class_2':2,'Class_3':3,'Class_4':4,'Class_5':5,'Class_6':6,'Class_7':7,'Class_8':8,'Class_9':9}).astype(int)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#x_train=train.drop(['target','id'],axis=1)
#y_train=train['target']
#x_test=test
#rf=RandomForestClassifier()
#rf.fit(x_train,y_train)
#rf.score(x_train,y_train)
#
#dt=DecisionTreeClassifier()
#dt.fit(x_train,y_train)
#pred=dt.predict(x_test)

y=X['target']
X=X.drop(['target'],axis=1)
X=X.drop(['id'],axis=1)
X.columns
#no id, no target


#split X and y into train and test sets
from sklearn.cross_validation import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)
#first we apply random forest without calibration
#using a bagging classifier
from sklearn.ensemble import BaggingClassifier
clf=RandomForestClassifier()
clfbag=BaggingClassifier(clf)
clfbag.fit(Xtrain,ytrain)
clfbag.score(Xtest,ytest)
ypreds=clfbag.predict_proba(Xtest)

from sklearn.metrics import log_loss
print("Loss without calibration :",log_loss(ytest,ypreds,normalize=True))
#0.6329

#now Randomforest with Calibration
from sklearn.calibration import CalibratedClassifierCV
clf=RandomForestClassifier()
calibclf=CalibratedClassifierCV(clf,method='isotonic')
calibclf.fit(Xtrain,ytrain)
calibclf.score(Xtest,ytest)
ypreds=calibclf.predict_proba(Xtest)
print("Loss with Calibration: ",log_loss(ytest,ypreds,normalize=True))
#0.5873
from sklearn.naive_bayes import GaussianNB
gaussian=GaussianNB()
gaussian.fit(Xtrain,ytrain)
gaussian.score(Xtest,ytest)
#61 %

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(Xtrain,ytrain)
dt.score(Xtest,ytest)
#71.1 %













