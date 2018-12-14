#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:10:05 2018


@author: JeremyKulchyk
"""

import scipy.fftpack
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn import preprocessing, cross_validation, neighbors, svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
np.set_printoptions(suppress=True) 



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
    # 5 = walking
df = df.replace('sitting', 1)
df = df.replace('standing', 2)
df = df.replace('sittingdown', 3)
df = df.replace('standingup', 4)
df = df.replace('walking', 5)
        
print(df.head())

df_norm = df[features]

# 4. Noise Reduction
# Changed -14420-11-2011 04:50:23.713 at (122078,Z4) to previous value (-142)

# Remove Gravity Component - high pass 0.25 HZ - http://www.tafpublications.com/gip_content/paper/JITDETS-1.1.1.pdf
# Remove Gravity Component - high pass 0.5 HZ - https://arxiv.org/pdf/1107.4414.pdf

from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 7
fs = 30       # sample rate, Hz
cutoff = 0.5 # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# Filter the data, and plot both the original and filtered signals.
df_norm_F = pd.DataFrame(butter_lowpass_filter(df_norm, cutoff, fs, order),columns=[features])


plt.subplot(2, 1, 2)
plt.plot(df_norm['x1'], 'b-', label='data')
plt.plot(df_norm_F['x1'], 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()



#df_denoiseGauss = df_norm.rolling(window=10000, win_type='gaussian', center=True).mean(std=0.5)


# Remove Noise
df_norm = df_norm_F.rolling(1000).mean()
#plt.plot(df_norm['x1'])

# Unity Normalization
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
#plt.plot(df_norm['x1'])

# Standarize
df_norm = (df_norm - df_norm.mean())/ df_norm.std()

# Segmentation - window size: 100, overlap: 80%
Window = 100
overlap = 0.80
step = int(Window - Window*0.80)

# Class 1 PreProcessing

df1 = df.loc[(df['class'] == 1)]
df1 = df1[features]
#df1 = df1.rolling(1000).mean().dropna()
df1 = (df1- df1.min()) / (df1.max() - df1.min())
df1 = (df1 - df1.mean())/ df1.std()
#df1 = pd.DataFrame(butter_lowpass_filter(df1, cutoff, fs, order),columns=[features])

Len = len(df1)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df1Seg = []
i = 0
while i < Len:
    df1Seg.append(df1.iloc[i:i+Window].values)
    i += step
    
# Class 2 Segmentation
df2 = df.loc[(df['class'] == 2)]
df2 = df2[features]
#df2 = df2.rolling(1000).mean().dropna()
df2 = (df2 - df2.min()) / (df2.max() - df2.min())
df2 = (df2 - df2.mean())/ df2.std()
#df2 = pd.DataFrame(butter_lowpass_filter(df2, cutoff, fs, order),columns=[features])

Len = len(df2)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df2Seg = []
i = 0
while i < Len:
    df2Seg.append(df2.iloc[i:i+Window].values)
    i += step
    
# Class 3 Segmentation
df3 = df.loc[(df['class'] == 3)]
df3 = df3[features]
#df3 = df3.rolling(1000).mean().dropna()
df3 = (df3 - df3.min()) / (df3.max() - df3.min())
df3 = (df3 - df3.mean())/ df3.std()
#df3 = pd.DataFrame(butter_lowpass_filter(df3, cutoff, fs, order),columns=[features])


Len = len(df3)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df3Seg = []
i = 0
while i < Len:
    df3Seg.append(df3.iloc[i:i+Window].values)
    i += step

# Class 4 Segmentation
df4 = df.loc[(df['class'] == 4)]
df4 = df4[features]
#df4 = df4.rolling(1000).mean().dropna()
df4 = (df4 - df4.min()) / (df4.max() - df4.min())
df4 = (df4 - df4.mean())/ df4.std()
#df4 = pd.DataFrame(butter_lowpass_filter(df4, cutoff, fs, order),columns=[features])

Len = len(df4)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df4Seg = []
i = 0
while i < Len:
    df4Seg.append(df4.iloc[i:i+Window].values)
    i += step

# Class 5 Segmentation
df5 = df.loc[(df['class'] == 5)]
df5 = df5[features]
#df5 = df5.rolling(1000).mean().dropna()
df5 = (df5 - df5.min()) / (df5.max() - df5.min())
df5 = (df5 - df5.mean())/ df5.std()
#df5 = pd.DataFrame(butter_lowpass_filter(df5, cutoff, fs, order),columns=[features])

Len = len(df5)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df5Seg = []
i = 0
while i < Len:
    df5Seg.append(df5.iloc[i:i+Window].values)
    i += step

    
print(len(df1),len(df1Seg))
print(len(df2),len(df2Seg))
print(len(df3),len(df3Seg))
print(len(df4),len(df4Seg))
print(len(df5),len(df5Seg))

print(len(df1Seg) + len(df2Seg) + len(df3Seg) + len(df4Seg) + len(df5Seg))

LENGTH = len(df1Seg) + len(df2Seg) + len(df3Seg) + len(df4Seg) + len(df5Seg)

"""
train_X = np.zeros((8004,100,12))
train_Y = np.zeros((8004))
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i]
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i]
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i]
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i]
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i]
    train_Y[j] = 4
    j += 1
    
print(df1Seg[0])
"""

df_sub1 = pd.DataFrame(data=df1Seg[0], index=[i for i in range(0,100)], columns=features)
print(df_sub1.head())

NewFeats = ['Mean_','Norm_','Var_','Std_','RMS_','Range_','Median_','Max_','Min_','Skew_','Kurt_']
NewFeatCols = []
for i in NewFeats:
    for j in features:
        NewFeatCols.append(i+j)
    
NewFeatCols.append('class')

df_feat = pd.DataFrame(0,columns=NewFeatCols,index=[str(i) for i in range(0,LENGTH)])

C = 0
for i in range(0,len(df1Seg)):
    df_sub1 = pd.DataFrame(data=df1Seg[i], index=[i for i in range(0,100)], columns=features)
    for f in features:
        df_feat.ix[str(C),'Mean_'+f] = round(df_sub1[f].mean(axis=0),2)
        df_feat.ix[str(C),'Norm_'+f] = round(np.sqrt(np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Var_'+f] = round(df_sub1[f].var(axis=0),2)
        df_feat.ix[str(C),'Std_'+f] = round(df_sub1[f].std(axis=0),2)
        df_feat.ix[str(C),'RMS_'+f] = round(np.sqrt((1/100)*np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Median_'+f] = round(df_sub1[f].median(axis=0),2)
        df_feat.ix[str(C),'Max_'+f] = round(df_sub1[f].max(axis=0),2)
        df_feat.ix[str(C),'Min_'+f] = round(df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Range_'+f] = round(df_sub1[f].max(axis=0) - df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Skew_'+f] = round(df_sub1[f].skew(axis=0),2)
        df_feat.ix[str(C),'Kurt_'+f] = round(df_sub1[f].kurtosis(axis=0),2)
        df_feat.ix[str(C),'class'] = 1
        
    C += 1
print(C)

    
for i in range(0,len(df2Seg)):
    df_sub1 = pd.DataFrame(data=df2Seg[i], index=[i for i in range(0,100)], columns=features)
    for f in features:
        df_feat.ix[str(C),'Mean_'+f] = round(df_sub1[f].mean(axis=0),2)
        df_feat.ix[str(C),'Norm_'+f] = round(np.sqrt(np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Var_'+f] = round(df_sub1[f].var(axis=0),2)
        df_feat.ix[str(C),'Std_'+f] = round(df_sub1[f].std(axis=0),2)
        df_feat.ix[str(C),'RMS_'+f] = round(np.sqrt((1/100)*np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Median_'+f] = round(df_sub1[f].median(axis=0),2)
        df_feat.ix[str(C),'Max_'+f] = round(df_sub1[f].max(axis=0),2)
        df_feat.ix[str(C),'Min_'+f] = round(df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Range_'+f] = round(df_sub1[f].max(axis=0) - df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Skew_'+f] = round(df_sub1[f].skew(axis=0),2)
        df_feat.ix[str(C),'Kurt_'+f] = round(df_sub1[f].kurtosis(axis=0),2)
        df_feat.ix[str(C),'class'] = 2
    C += 1
print(C)

for i in range(0,len(df3Seg)):
    df_sub1 = pd.DataFrame(data=df3Seg[i], index=[i for i in range(0,100)], columns=features)
    for f in features:
        df_feat.ix[str(C),'Mean_'+f] = round(df_sub1[f].mean(axis=0),2)
        df_feat.ix[str(C),'Norm_'+f] = round(np.sqrt(np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Var_'+f] = round(df_sub1[f].var(axis=0),2)
        df_feat.ix[str(C),'Std_'+f] = round(df_sub1[f].std(axis=0),2)
        df_feat.ix[str(C),'RMS_'+f] = round(np.sqrt((1/100)*np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Median_'+f] = round(df_sub1[f].median(axis=0),2)
        df_feat.ix[str(C),'Max_'+f] = round(df_sub1[f].max(axis=0),2)
        df_feat.ix[str(C),'Min_'+f] = round(df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Range_'+f] = round(df_sub1[f].max(axis=0) - df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Skew_'+f] = round(df_sub1[f].skew(axis=0),2)
        df_feat.ix[str(C),'Kurt_'+f] = round(df_sub1[f].kurtosis(axis=0),2)
        df_feat.ix[str(C),'class'] = 3
    C += 1
print(C)
    
for i in range(0,len(df4Seg)):
    df_sub1 = pd.DataFrame(data=df4Seg[i], index=[i for i in range(0,100)], columns=features)
    for f in features:
        df_feat.ix[str(C),'Mean_'+f] = round(df_sub1[f].mean(axis=0),2)
        df_feat.ix[str(C),'Norm_'+f] = round(np.sqrt(np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Var_'+f] = round(df_sub1[f].var(axis=0),2)
        df_feat.ix[str(C),'Std_'+f] = round(df_sub1[f].std(axis=0),2)
        df_feat.ix[str(C),'RMS_'+f] = round(np.sqrt((1/100)*np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Median_'+f] = round(df_sub1[f].median(axis=0),2)
        df_feat.ix[str(C),'Max_'+f] = round(df_sub1[f].max(axis=0),2)
        df_feat.ix[str(C),'Min_'+f] = round(df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Range_'+f] = round(df_sub1[f].max(axis=0) - df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Skew_'+f] = round(df_sub1[f].skew(axis=0),2)
        df_feat.ix[str(C),'Kurt_'+f] = round(df_sub1[f].kurtosis(axis=0),2)
        df_feat.ix[str(C),'class'] = 4
    C += 1
print(C)
    
for i in range(0,len(df5Seg)):
    df_sub1 = pd.DataFrame(data=df5Seg[i], index=[i for i in range(0,100)], columns=features)
    for f in features:
        df_feat.ix[str(C),'Mean_'+f] = round(df_sub1[f].mean(axis=0),2)
        df_feat.ix[str(C),'Norm_'+f] = round(np.sqrt(np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Var_'+f] = round(df_sub1[f].var(axis=0),2)
        df_feat.ix[str(C),'Std_'+f] = round(df_sub1[f].std(axis=0),2)
        df_feat.ix[str(C),'RMS_'+f] = round(np.sqrt((1/100)*np.square(df_sub1[f]).sum(axis=0)),2)
        df_feat.ix[str(C),'Median_'+f] = round(df_sub1[f].median(axis=0),2)
        df_feat.ix[str(C),'Max_'+f] = round(df_sub1[f].max(axis=0),2)
        df_feat.ix[str(C),'Min_'+f] = round(df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Range_'+f] = round(df_sub1[f].max(axis=0) - df_sub1[f].min(axis=0),2)
        df_feat.ix[str(C),'Skew_'+f] = round(df_sub1[f].skew(axis=0),2)
        df_feat.ix[str(C),'Kurt_'+f] = round(df_sub1[f].kurtosis(axis=0),2)
        df_feat.ix[str(C),'class'] = 5
    C += 1
print(C)

print(df_feat.head())

df_feat.to_csv("NewFeats.csv", encoding='utf-8', index=False)
"""
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i]
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i]
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i]
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i]
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i]
    train_Y[j] = 4
    j += 1
    
print(df1Seg[0])
"""

 

#df = pd.concat([df1,df2,df3,df4,df5])

#print(len(df))


# FEATURE SELECTION


"""
# load data
features = list(df.columns.values)
del features[len(features)-1]
X = df.loc[:, features].values
Y = df.loc[:,['class']].values.ravel()

FeatImp = {}
for i in range(0,100):
    # feature extractionq
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    #print(model.feature_importances_)
    RankedFeats = []
    count = 0
    for Rank in model.feature_importances_:
        try:
            FeatImp[features[count]] += Rank
        except:
            FeatImp[features[count]] = Rank
        count += 1
        RankedFeats.append(Rank)
for key in FeatImp:
    FeatImp[key] = FeatImp[key]/100
    
print(FeatImp)
"""
ExcelFile = "NewFeats.csv"

df_feat = pd.read_csv(ExcelFile)

#print(df_feat.tail())

# Dimensionality Reduction
features = list(df_feat.columns.values)
del features[len(features)-1]
x = df_feat.loc[:, features].values
y = df_feat.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

#print(len(x),len(y))

#print(pd.DataFrame(data = x, columns = features).head())

pca = PCA(n_components=23)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23'])


print(pca.explained_variance_ratio_)

j = 0
for x in pca.explained_variance_ratio_:
    j = j + x
print(j)

principalDf['class'] = df_feat['class']
print(principalDf.tail())

df_norm = principalDf

features = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21','PC22','PC23']

testsize = 0.3

#features = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3','x4', 'y4', 'z4']

# Gaussian Fit
X = np.array(df_norm[features])
y = np.array(df_norm['class'].values.ravel())
seed = 7

print(X[0])
print(y)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
ScoresDict = {"NB":{},"KNN":{},"DT":{},"E1":{},"E2":{},"RF":{}}

Count = 0
Name = "NB"
for train, test in kfold.split(X, y):
    clf = GaussianNB()
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")


Count = 0
Name = "KNN"
for train, test in kfold.split(X, y):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")


Count = 0
Name = "DT"
for train, test in kfold.split(X, y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")

"""

Count = 0
Name = "E1"
for train, test in kfold.split(X, y):
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model3))
    clf = VotingClassifier(estimators)
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")
"""


Count = 0
Name = "E2"
for train, test in kfold.split(X, y):
    # create the sub models
    estimators = []
    model1 = neighbors.KNeighborsClassifier()
    estimators.append(('knn', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = GaussianNB()
    estimators.append(('gnb', model3))
    clf = VotingClassifier(estimators)
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")

Count = 0
Name = "RF"
for train, test in kfold.split(X, y):
    clf = RandomForestRegressor(n_estimators = 10, random_state = 42)
    clf.fit(X[train], y[train])
    y_test = y[test]
    y_pred = clf.predict(X[test])
    y_pred = [int(x) for x in y_pred]
    CM = confusion_matrix(y_test, y_pred)
    Acc = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Prec = precision_score(y_test, y_pred, average="macro")
    Rec = recall_score(y_test, y_pred, average="macro")
    if Count == 0:
        ScoresDict[Name]["CM"] = CM
        ScoresDict[Name]["Acc"] = Acc
        ScoresDict[Name]["F1"] = F1
        ScoresDict[Name]["Prec"] = Prec
        ScoresDict[Name]["Rec"] = Rec
    else:
        ScoresDict[Name]["CM"].__add__(CM)
        ScoresDict[Name]["Acc"] += Acc
        ScoresDict[Name]["F1"] += F1
        ScoresDict[Name]["Prec"] += Prec
        ScoresDict[Name]["Rec"] += Rec
    Count += 1
ScoresDict[Name]["CM"] = np.divide(ScoresDict[Name]["CM"], 10).round(2)
ScoresDict[Name]["Acc"] = (ScoresDict[Name]["Acc"]/10).round(4)
ScoresDict[Name]["F1"] = (ScoresDict[Name]["F1"]/10).round(4)
ScoresDict[Name]["Prec"] = (ScoresDict[Name]["Prec"]/10).round(4)
ScoresDict[Name]["Rec"] = (ScoresDict[Name]["Rec"]/10).round(4)

print(Name)
print(ScoresDict[Name])
print(" ")


"""
# Unity Normalization
df_norm = (df[features]- df[features].min()) / (df[features].max() - df[features].min())

plt.plot(df_norm['x1'])


#plt.plot(df_norm[['x1','x2','x3','x4']])
"""

# Feature Importance


"""
# load data
array = df_norm.values
features = ['age', 'how_tall_in_meters', 'weight',
       'body_mass_index', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3',
       'x4', 'y4', 'z4']
X = df_norm.loc[:, features].values

Y = df.loc[:,['class']].values.ravel()

FeatImp = {}
for i in range(0,100):
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    #print(model.feature_importances_)
    RankedFeats = []
    count = 0
    for Rank in model.feature_importances_:
        try:
            FeatImp[features[count]] += Rank
        except:
            FeatImp[features[count]] = Rank
        count += 1
        RankedFeats.append(Rank)
for key in FeatImp:
    FeatImp[key] = FeatImp[key]/100
    
print(FeatImp)
    
'z2':               0.15673764845147908, 
'y2':               0.12995935205198866, 
'z1':               0.09814524663644335, 
'x4':               0.08748674664001577, 
'y1':               0.08598603660456064, 
'y3':               0.06663073382763031, 
'x2':               0.06250908066624497, 
'z4':               0.05422075751972119,
'y4':               0.05352471474673695, 
'z3':               0.04915935733149239, 
85%^

'x1':               0.04827884995921617, 
'x3':               0.03550913586125243, 
'how_tall':         0.020270970172990555, 
'weight':           0.02023131775122682, 
'body_mass_index':  0.01808484182340371, 
'age':              0.013265209955597127, 
"""
# Update df_norm array
#features = ['y1', 'z1', 'x2', 'y2', 'z2', 'y3', 'z3','x4', 'y4', 'z4']
#features = ['z1', 'z2','z3','z4']
#df_norm = df_norm[features]


"""
Take FFT and find noise
Do Feature ex




# Dimensionality Reduction
features = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3',
       'x4', 'y4', 'z4']

x = df_norm.values
y = df.loc[:,['class']].values

x = StandardScaler().fit_transform(x)

#print(pd.DataFrame(data = x, columns = features).head())

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

df_norm = pd.concat([principalDf, df[['class']]], axis = 1)
#rint(df_norm.head())

# Classifiers
"""
# Naive Bayes
"""


# k-Nearest Neighbour
X = np.array(df_norm)
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print("KNN:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
example_measures = np.array(X_train)
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
#print(prediction)
#print(y_train)
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")



# Decision Tree
X = np.array(df_norm)
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print("DT:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
example_measures = np.array(X_train)
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
#print(prediction)
#print(y_train)
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")



# Ensemble Methods

# Voting Ensemble - Logistic Regression, Decision Tree, SVM
X = np.array(df_norm)
y = np.array(df['class'])
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = VotingClassifier(estimators)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Ensemble 1:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")



# Voting Ensemble - Logistic Regression, Decision Tree, SVM
X = np.array(df_norm)
y = np.array(df['class'])
# create the sub models
estimators = []
model1 = neighbors.KNeighborsClassifier()
estimators.append(('knn', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = GaussianNB()
estimators.append(('gnb', model3))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = VotingClassifier(estimators)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Ensemble 2:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")


# Random Forest
from sklearn.ensemble import RandomForestRegressor

X = np.array(df_norm)
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Random Forest:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")

# AdaBoost Classification

# Regularization


# For Each Classifier, Calculate
    # Accuracy
    # Precision
    # Recall
    # ROC Curve

# Validation Strategies
    # K-fold cross validation
    # Leave one out

"""