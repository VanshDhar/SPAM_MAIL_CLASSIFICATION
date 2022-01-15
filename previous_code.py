#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:34:23 2021

@author: vanshdhar
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, KFold
import random

counter=0
#IMPORTING DATASET

dataset=pd.read_csv('spambase_data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#REPLACING MISSING VALUES
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)
#X = random.shuffle(X)
#SPLITTING DATASET INTO TRAINING AND TESTING
"""
start_index = [0,920,1840,2760,3680]
stop_index = [920,1840,2760,3680,4601]
#X_1fold=X[:920,:]
#X_2fold=X[920:1840,:]
#X_3fold=X[1840:2760,:]
#X_4fold=X[2760:3680,:]
#X_5fold=X[3680:,:]

#Y_1fold=Y[:920]
#Y_2fold=Y[920:1840]
#Y_3fold=Y[1840:2760]
#Y_4fold=Y[2760:3680]
#Y_5fold=Y[3680:]

def fold_generator(X, Y, start, stop):
    #X_train = []
    #Y_train = []
    X_test = X[start:stop]
    Y_test = Y[start:stop]
    if (start>0):
        X_train= X[:start]
        Y_train=Y[:start]
    else:
        X_train= X[stop:]
        Y_train=Y[stop:]
        
    if (stop<4601)and(start>0):
        X_train = np.append((X_train,X[stop:]),axis=0)
        Y_train = np.append((Y_train,Y[stop:]),axis=0)
        
    X_train.reshape(3681,57)
    Y_train.reshape(3681)
    return X_train, Y_train, X_test, Y_test
"""
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=1)


#FEATURE SCALING
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#X_train, Y_train, X_test, Y_test = fold_generator(X,Y,920,1840)
Classification_Algorithms = ['Logistic_Regression', 'K_Nearest_Neighbors', 'Support_Vector_Machine', 'Naive_Bayes', 'Random_Forest']
"""
for start,stop in zip(start_index,stop_index):
    counter+=1
    #print(str(stop))
    #X_train, Y_train, X_test, Y_test = fold_generator(X,Y,start,stop)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #X_train.shape()
    print('Fold no: '+str(counter))
    for ca in Classification_Algorithms:
        print('\n')
        print(ca)
        if (ca=='Logistic_Regression'):
            classifier = LogisticRegression(random_state=1)
        elif (ca=='K_Nearest_Neighbors'):
            classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
        elif (ca=='Support_Vector_Machine'):
            classifier = SVC(kernel = 'rbf', random_state=1)
        elif (ca=='Naive_Bayes'):
            classifier = GaussianNB()
        elif (ca=='Random_Forest'):
            classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=1)
            
        classifier.fit(X_train, Y_train)
    
        print('Algorith: '+ca)
        #accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
        kfold = KFold(n_splits=5, shuffle=True, random_state=90210)
        #accuracies = cross_validate(estimator = classifier, X = X_train, y = Y_train, cv = kfold, scoring=['false_positive_rate','false_negative_rate','overall_error_rate'])
        #print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        #print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    
        #PREDICTING THE RESULTS OF TEST SET
        #y_pred = classifier.predict(X_test)
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
        #cm = confusion_matrix(Y_test, y_pred)
        #print(cm)
        #ac = accuracy_score(Y_test, y_pred)
        #print(ac)
        #tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        #print('false_positive_rate: '+str(fp/(fp+tn)))
        #print('false_negative_rate: '+str(fn/(fn+tp)))
        #print('overall_error_rate: '+str((fp+fn)/(tn+fp+fn+tp)))
"""
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
for train_index, test_index in kfold.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #X_train.shape()

    for ca in Classification_Algorithms:
            print('\n')
            print(ca)
            if (ca=='Logistic_Regression'):
                classifier = LogisticRegression(random_state=1)
            elif (ca=='K_Nearest_Neighbors'):
                classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
            elif (ca=='Support_Vector_Machine'):
                classifier = SVC(kernel = 'rbf', random_state=1)
            elif (ca=='Naive_Bayes'):
                classifier = GaussianNB()
            elif (ca=='Random_Forest'):
                classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state=1)
            
            
            classifier.fit(X_train, Y_train)
        
            print('Algorith: '+ca)
            #accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
            
            #accuracies = cross_validate(estimator = classifier, X = X_train, y = Y_train, cv = kfold, scoring=['false_positive_rate','false_negative_rate','overall_error_rate'])
            #print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
            #print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
        
            #PREDICTING THE RESULTS OF TEST SET
            y_pred = classifier.predict(X_test)
            #print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
            cm = confusion_matrix(Y_test, y_pred)
            print(cm)
            #ac = accuracy_score(Y_test, y_pred)
            #print(ac)
            tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
            print('false_positive_rate: '+str(fp/(fp+tn)))
            print('false_negative_rate: '+str(fn/(fn+tp)))
            print('overall_error_rate: '+str((fp+fn)/(tn+fp+fn+tp)))

