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

from sklearn.linear_model import LogisticRegression

train_dataf = pd.read_csv('C:/Users/Ornello/Documents/TitanicSurvivalPrediction/train.csv')

print(train_dataf.info())
#Cabin and Age are not complete data sets
# Age missing ~270 values, Cabin missing ~ 700 values

print(train_dataf.describe(include = ['O']))  #'0' for non-numeric data
#Data set only includes 891/2240 passengers on the titanic
#38% of passengers included in the data set survived
#mostly male: 577/891
#Some duplicate tickets? only 691/891 unique
#Most passengers in data set embarked from 'S': 644/889
