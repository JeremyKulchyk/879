#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:10:05 2018


CNN0

3 different ensembles:
    1. 4 weak CNN's with each sensor
    2. 3 weak CNN's with each component
    3. x groups of highest correlated signals




@author: JeremyKulchyk
"""

import scipy.fftpack
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing, cross_validation, neighbors, svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

ExcelFile = "/Users/JeremyKulchyk/Downloads/879Project/dataset.csv"

df = pd.read_csv(ExcelFile)




""" Feature Extraction  """
testsize = 0.3

features = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3','x4', 'y4', 'z4']


# PRE PROCESSING
# 1. Formatted Excel Sheet

# 2. Get rid of Gender and names columns
#df = df[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3','x4', 'y4', 'z4','class']]

# 3. Change classes to numbers:
    # 1 = sitting
    # 2 = standing
    # 3 = sitting down
    # 4 = standing up
    # 5 = walkings
df = df.replace('sitting', 1)
df = df.replace('standing', 2)
df = df.replace('sittingdown', 3)
df = df.replace('standingup', 4)
df = df.replace('walking', 5)
        
  
with open('Results2.txt','r') as inf:
    Dict = eval(inf.read())
#print(Dict)

RankedList = []

for keys in Dict.keys():
    NewDict = {'Name':keys,
               'AvgCor':Dict[keys]['AvgCor'],
               'Accuraccy':Dict[keys]['Accuraccy'],
               'F1':Dict[keys]['F1'],
               'Precision':Dict[keys]['Precision'],
               'Recall':Dict[keys]['Recall'],
               'Components':Dict[keys]['Components'],
               'CM':Dict[keys]['CM']}
    RankedList.append(NewDict)

#print(RankedList)

newlist = sorted(RankedList, key=lambda k: k['Accuraccy']) 

print(newlist[217])
print(newlist[218])
print(newlist[219])


#print(newlist)

X = []
F1 = []
Acc = []
Prec = []
Rec = []
for elem in newlist:
    X.append(elem['AvgCor'])
    F1.append(elem['F1'])
    Acc.append(elem['Accuraccy'])
    Prec.append(elem['Precision'])
    Rec.append(elem['Recall'])


plt.plot(X,Acc)




