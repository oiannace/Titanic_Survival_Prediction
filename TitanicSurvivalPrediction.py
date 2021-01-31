# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:03:54 2021

@author: Ornello
"""

import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

train_dataf = pd.read_csv('C:/Users/Ornello/Documents/PersonalProjects/TitanicSurvivalPrediction/train.csv')
test_dataf = pd.read_csv('C:/Users/Ornello/Documents/PersonalProjects/TitanicSurvivalPrediction/test.csv')
#Remove Survived column from x axis training data
y_train = train_dataf['Survived']

#Need to create larger groupings for Age and Fare
#Determine if the current fields show any correlation, if not possibly create more signifcant features


for i in range(len(train_dataf.index)):
    if train_dataf.loc[i, 'Sex'] == 'male':
        train_dataf.loc[i, 'Sex'] = 0
    elif train_dataf.loc[i, 'Sex'] == 'female':
        train_dataf.loc[i, 'Sex'] = 1

for i in range(len(test_dataf.index)):
    if test_dataf.loc[i, 'Sex'] == 'male':
        test_dataf.loc[i, 'Sex'] = 0
    elif test_dataf.loc[i, 'Sex'] == 'female':
        test_dataf.loc[i, 'Sex'] = 1
        
#Pandas syntax to do above in one line
#train_dataf['Sex'] = train_dataf['Sex'].map({'female': 1, 'male': 0}).astype(int)

#Fill missing values in Embarked field with most frequent occurence ('S")
train_dataf['Embarked'] = train_dataf['Embarked'].fillna(train_dataf['Embarked'].mode()[0])
test_dataf['Embarked'] = test_dataf['Embarked'].fillna(test_dataf['Embarked'].mode()[0])

train_dataf['Embarked'] = train_dataf['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_dataf['Embarked'] = test_dataf['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_dataf = train_dataf.drop(columns = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin','Fare' ,'Age'])
test_dataf = test_dataf.drop(columns = ['PassengerId','Name', 'Cabin', 'Ticket', 'Age', 'Fare'])


GNaiveBayes = GaussianNB()
GNaiveBayes.fit(train_dataf, y_train)
y_prediction = GNaiveBayes.fit(train_dataf, y_train).predict(test_dataf)
accuracyGNB = round(GNaiveBayes.score(train_dataf, y_train)*100, 2)
print(accuracyGNB)
#79.01

#Cabin and Age are not complete data sets
# Age missing ~270 values, Cabin missing ~ 700 values
#train_dataf = train_dataf.drop(columns = ['Cabin', 'Age'])

#print(train_dataf.head(20))
#print(train_dataf.describe(include = ['O']))  #'0' for non-numeric data
#Data set only includes 891/2240 passengers on the titanic
#38% of passengers included in the data set survived
#mostly male: 577/891
#Some duplicate tickets? only 691/891 unique
#Most passengers in data set embarked from 'S': 644/889
