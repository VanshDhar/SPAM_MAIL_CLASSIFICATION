#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 23:34:23 2021

@author: vanshdhar
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#TO CHECK THE OVERALL PERFORMANCE OF EACH CLASSIFICATION MODEL 
average_overall_error_rate=[0,0,0,0,0]
average_false_positive_rate=[0,0,0,0,0]
average_false_negative_rate=[0,0,0,0,0]
#IMPORTING DATASET
dataset=pd.read_csv('spambase_data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#REPLACING MISSING VALUES
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)

#LIST OF ALL CLASSIFICATION ALGORITHMS TO BE USED
Classification_Algorithms = ['Logistic_Regression', 'K_Nearest_Neighbors', 'Support_Vector_Machine', 'Naive_Bayes', 'Random_Forest']

#GENERATING A KFOLD SPLIT OBJECT
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
i=0
for ca in Classification_Algorithms:
        #KEEPING COUNTER TO KNOW WHICH ITERATION OF THE KFOLD SPLIT WE ARE ON
        counter=0
        print('\n')
        #print(ca)
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
        
        for train_index, test_index in kfold.split(X):
            print('\n')
            
            counter+=1
            #print('Fold no: '+str(counter))
            #Splitting the dataset
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            #FEATURE SCALING
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        
            #TRAINING THE CLASSIFIER        
            classifier.fit(X_train, Y_train)
        
        
            #PREDICTING THE RESULTS OF TEST SET
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(Y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
            print('Algorithm: '+ca+' | Fold no: '+str(counter)+' | false_positive_rate: '+str(fp/(fp+tn))+' | false_negative_rate: '+str(fn/(fn+tp))+' | overall_error_rate: '+str((fp+fn)/(tn+fp+fn+tp)))
            average_overall_error_rate[i] += (fp+fn)/(tn+fp+fn+tp)
            average_false_positive_rate[i] += fp/(fp+tn)
            average_false_negative_rate[i] += fn/(fn+tp)
            
        average_overall_error_rate[i] /=5
        average_false_positive_rate[i] /=5
        average_false_negative_rate[i] /=5
        i+=1

print('\n')        
min_value = min(average_overall_error_rate)
min_index = average_overall_error_rate.index(min_value)
print('Best Classification Algorithm: '+Classification_Algorithms[min_index]+' | Average Overall Error Rate: '+str(average_overall_error_rate[min_index])+' | Average False Positive Rate: '+str(average_false_positive_rate[min_index])+' | Average False Negative Rate: '+str(average_false_negative_rate[min_index]))
#print('Average Overall Error Rate: '+str(average_overall_error_rate[min_index]))
    



