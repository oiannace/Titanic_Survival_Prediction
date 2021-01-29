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

#Need to transfer String data to numeric: Sex, Embarked
#Make fare into catagories and assign levels (eg. 1,2,3)
#possibly create more relevant features from current fields?

train_dataf = train_dataf.drop(columns = ['Survived', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked','Fare' ,'Age'])
test_dataf = test_dataf.drop(columns = ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket', 'Age', 'Fare'])

train_dataf.shape
y_train.shape
test_dataf.shape

GNaiveBayes = GaussianNB()
#GNaiveBayes.fit(train_dataf, y_train)
y_prediction = GNaiveBayes.fit(train_dataf, y_train).predict(test_dataf)
accuracyGNB = round(GNaiveBayes.score(train_dataf, y_train)*100, 2)
print(accuracyGNB)
#66.89

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
