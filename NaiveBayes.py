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
#df1 = (df1- df1.min()) / (df1.max() - df1.min())
#df1 = (df1 - df1.mean())/ df1.std()
#df1 = pd.DataFrame(butter_lowpass_filter(df1, cutoff, fs, order),columns=[features])

# Class 1 Feature Extraction
# Mean
df1['Mean1'] = df1[['x1','y1','z1']].mean(axis=1)
df1['Mean2'] = df1[['x2','y2','z2']].mean(axis=1)
df1['Mean3'] = df1[['x3','y3','z3']].mean(axis=1)
df1['Mean4'] = df1[['x4','y4','z4']].mean(axis=1)
df1['Mean'] = df1[features].mean(axis=1)

# Norms
df1['Norm1'] = np.sqrt(np.square(df1[['x1','y1','z1']]).sum(axis=1))
df1['Norm2'] = np.sqrt(np.square(df1[['x2','y2','z2']]).sum(axis=1))
df1['Norm3'] = np.sqrt(np.square(df1[['x3','y3','z3']]).sum(axis=1))
df1['Norm4'] = np.sqrt(np.square(df1[['x4','y4','z4']]).sum(axis=1))
df1['Norm'] = np.sqrt(np.square(df1[features]).sum(axis=1))

# RMS
df1['RMS1'] = np.sqrt((1/3)*np.square(df1[['x1','y1','z1']]).sum(axis=1))
df1['RMS2'] = np.sqrt((1/3)*np.square(df1[['x2','y2','z2']]).sum(axis=1))
df1['RMS3'] = np.sqrt((1/3)*np.square(df1[['x3','y3','z3']]).sum(axis=1))
df1['RMS4'] = np.sqrt((1/3)*np.square(df1[['x4','y4','z4']]).sum(axis=1))
df1['RMS'] = np.sqrt((1/12)*np.square(df1[features]).sum(axis=1))

# Variance
df1['Var1'] = df1[['x1','y1','z1']].var(axis=1)
df1['Var2'] = df1[['x2','y2','z2']].var(axis=1)
df1['Var3'] = df1[['x3','y3','z3']].var(axis=1)
df1['Var4'] = df1[['x4','y4','z4']].var(axis=1)
df1['Var'] = df1[features].var(axis=1)

# Std
df1['Std1'] = df1[['x1','y1','z1']].std(axis=1)
df1['Std2'] = df1[['x2','y2','z2']].std(axis=1)
df1['Std3'] = df1[['x3','y3','z3']].std(axis=1)
df1['Std4'] = df1[['x4','y4','z4']].std(axis=1)
df1['Std'] = df1[features].std(axis=1)

# differences
df1[['DX1','DY1','DZ1','DX2','DY2','DZ2','DX3','DY3','DZ3','DX4','DY4','DZ4']] = df1[features].diff()
df1 = df1.dropna()

# Median
df1['Med1'] = df1[['x1','y1','z1']].median(axis=1)
df1['Med2'] = df1[['x2','y2','z2']].median(axis=1)
df1['Med3'] = df1[['x3','y3','z3']].median(axis=1)
df1['Med4'] = df1[['x4','y4','z4']].median(axis=1)
df1['Med'] = df1[features].median(axis=1)

# Max
df1['Max1'] = df1[['x1','y1','z1']].max(axis=1)
df1['Max2'] = df1[['x2','y2','z2']].max(axis=1)
df1['Max3'] = df1[['x3','y3','z3']].max(axis=1)
df1['Max4'] = df1[['x4','y4','z4']].max(axis=1)
df1['Max'] = df1[features].max(axis=1)

# Min
df1['Min1'] = df1[['x1','y1','z1']].min(axis=1)
df1['Min2'] = df1[['x2','y2','z2']].min(axis=1)
df1['Min3'] = df1[['x3','y3','z3']].min(axis=1)
df1['Min4'] = df1[['x4','y4','z4']].min(axis=1)
df1['Min'] = df1[features].min(axis=1)

# Skewness
df1['Skew1'] = df1[['x1','y1','z1']].skew(axis=1)
df1['Skew2'] = df1[['x2','y2','z2']].skew(axis=1)
df1['Skew3'] = df1[['x3','y3','z3']].skew(axis=1)
df1['Skew4'] = df1[['x4','y4','z4']].skew(axis=1)
df1['Skew'] = df1[features].skew(axis=1)

# Kurtosis
df1['Kurt'] = df1[features].kurtosis(axis=1)

df1['class'] = 1





# Class 2 PreProcessing


df2 = df.loc[(df['class'] == 2)]
df2 = df2[features]
#df2 = df2.rolling(1000).mean().dropna()
#df2 = (df2- df2.min()) / (df2.max() - df2.min())
#df2 = (df2 - df2.mean())/ df2.std()
#df2 = pd.DataFrame(butter_lowpass_filter(df2, cutoff, fs, order),columns=[features])


# Class 2 Feature Extraction
# Mean
df2['Mean1'] = df2[['x1','y1','z1']].mean(axis=1)
df2['Mean2'] = df2[['x2','y2','z2']].mean(axis=1)
df2['Mean3'] = df2[['x3','y3','z3']].mean(axis=1)
df2['Mean4'] = df2[['x4','y4','z4']].mean(axis=1)
df2['Mean'] = df2[features].mean(axis=1)


# Norms
df2['Norm1'] = np.sqrt(np.square(df2[['x1','y1','z1']]).sum(axis=1))
df2['Norm2'] = np.sqrt(np.square(df2[['x2','y2','z2']]).sum(axis=1))
df2['Norm3'] = np.sqrt(np.square(df2[['x3','y3','z3']]).sum(axis=1))
df2['Norm4'] = np.sqrt(np.square(df2[['x4','y4','z4']]).sum(axis=1))
df2['Norm'] = np.sqrt(np.square(df2[features]).sum(axis=1))


# RMS
df2['RMS1'] = np.sqrt((1/3)*np.square(df2[['x1','y1','z1']]).sum(axis=1))
df2['RMS2'] = np.sqrt((1/3)*np.square(df2[['x2','y2','z2']]).sum(axis=1))
df2['RMS3'] = np.sqrt((1/3)*np.square(df2[['x3','y3','z3']]).sum(axis=1))
df2['RMS4'] = np.sqrt((1/3)*np.square(df2[['x4','y4','z4']]).sum(axis=1))
df2['RMS'] = np.sqrt((1/12)*np.square(df2[features]).sum(axis=1))


# Variance
df2['Var1'] = df2[['x1','y1','z1']].var(axis=1)
df2['Var2'] = df2[['x2','y2','z2']].var(axis=1)
df2['Var3'] = df2[['x3','y3','z3']].var(axis=1)
df2['Var4'] = df2[['x4','y4','z4']].var(axis=1)
df2['Var'] = df2[features].var(axis=1)


# Std
df2['Std1'] = df2[['x1','y1','z1']].std(axis=1)
df2['Std2'] = df2[['x2','y2','z2']].std(axis=1)
df2['Std3'] = df2[['x3','y3','z3']].std(axis=1)
df2['Std4'] = df2[['x4','y4','z4']].std(axis=1)
df2['Std'] = df2[features].std(axis=1)


# differences
df2[['DX1','DY1','DZ1','DX2','DY2','DZ2','DX3','DY3','DZ3','DX4','DY4','DZ4']] = df2[features].diff()
df2 = df2.dropna()


# Median
df2['Med1'] = df2[['x1','y1','z1']].median(axis=1)
df2['Med2'] = df2[['x2','y2','z2']].median(axis=1)
df2['Med3'] = df2[['x3','y3','z3']].median(axis=1)
df2['Med4'] = df2[['x4','y4','z4']].median(axis=1)
df2['Med'] = df2[features].median(axis=1)


# Max
df2['Max1'] = df2[['x1','y1','z1']].max(axis=1)
df2['Max2'] = df2[['x2','y2','z2']].max(axis=1)
df2['Max3'] = df2[['x3','y3','z3']].max(axis=1)
df2['Max4'] = df2[['x4','y4','z4']].max(axis=1)
df2['Max'] = df2[features].max(axis=1)


# Min
df2['Min1'] = df2[['x1','y1','z1']].min(axis=1)
df2['Min2'] = df2[['x2','y2','z2']].min(axis=1)
df2['Min3'] = df2[['x3','y3','z3']].min(axis=1)
df2['Min4'] = df2[['x4','y4','z4']].min(axis=1)
df2['Min'] = df2[features].min(axis=1)


# Skewness
df2['Skew1'] = df2[['x1','y1','z1']].skew(axis=1)
df2['Skew2'] = df2[['x2','y2','z2']].skew(axis=1)
df2['Skew3'] = df2[['x3','y3','z3']].skew(axis=1)
df2['Skew4'] = df2[['x4','y4','z4']].skew(axis=1)
df2['Skew'] = df2[features].skew(axis=1)


# Kurtosis
df2['Kurt'] = df2[features].kurtosis(axis=1)


df2['class'] = 2







# Class 3 PreProcessing


df3 = df.loc[(df['class'] == 3)]
df3 = df3[features]
#df3 = df3.rolling(1000).mean().dropna()
#df3 = (df3- df3.min()) / (df3.max() - df3.min())
#df3 = (df3 - df3.mean())/ df3.std()
#df3 = pd.DataFrame(butter_lowpass_filter(df3, cutoff, fs, order),columns=[features])


# Class 3 Feature Extraction
# Mean
df3['Mean1'] = df3[['x1','y1','z1']].mean(axis=1)
df3['Mean2'] = df3[['x2','y2','z2']].mean(axis=1)
df3['Mean3'] = df3[['x3','y3','z3']].mean(axis=1)
df3['Mean4'] = df3[['x4','y4','z4']].mean(axis=1)
df3['Mean'] = df3[features].mean(axis=1)


# Norms
df3['Norm1'] = np.sqrt(np.square(df3[['x1','y1','z1']]).sum(axis=1))
df3['Norm2'] = np.sqrt(np.square(df3[['x2','y2','z2']]).sum(axis=1))
df3['Norm3'] = np.sqrt(np.square(df3[['x3','y3','z3']]).sum(axis=1))
df3['Norm4'] = np.sqrt(np.square(df3[['x4','y4','z4']]).sum(axis=1))
df3['Norm'] = np.sqrt(np.square(df3[features]).sum(axis=1))


# RMS
df3['RMS1'] = np.sqrt((1/3)*np.square(df3[['x1','y1','z1']]).sum(axis=1))
df3['RMS2'] = np.sqrt((1/3)*np.square(df3[['x2','y2','z2']]).sum(axis=1))
df3['RMS3'] = np.sqrt((1/3)*np.square(df3[['x3','y3','z3']]).sum(axis=1))
df3['RMS4'] = np.sqrt((1/3)*np.square(df3[['x4','y4','z4']]).sum(axis=1))
df3['RMS'] = np.sqrt((1/12)*np.square(df3[features]).sum(axis=1))


# Variance
df3['Var1'] = df3[['x1','y1','z1']].var(axis=1)
df3['Var2'] = df3[['x2','y2','z2']].var(axis=1)
df3['Var3'] = df3[['x3','y3','z3']].var(axis=1)
df3['Var4'] = df3[['x4','y4','z4']].var(axis=1)
df3['Var'] = df3[features].var(axis=1)


# Std
df3['Std1'] = df3[['x1','y1','z1']].std(axis=1)
df3['Std2'] = df3[['x2','y2','z2']].std(axis=1)
df3['Std3'] = df3[['x3','y3','z3']].std(axis=1)
df3['Std4'] = df3[['x4','y4','z4']].std(axis=1)
df3['Std'] = df3[features].std(axis=1)


# differences
df3[['DX1','DY1','DZ1','DX2','DY2','DZ2','DX3','DY3','DZ3','DX4','DY4','DZ4']] = df3[features].diff()
df3 = df3.dropna()


# Median
df3['Med1'] = df3[['x1','y1','z1']].median(axis=1)
df3['Med2'] = df3[['x2','y2','z2']].median(axis=1)
df3['Med3'] = df3[['x3','y3','z3']].median(axis=1)
df3['Med4'] = df3[['x4','y4','z4']].median(axis=1)
df3['Med'] = df3[features].median(axis=1)


# Max
df3['Max1'] = df3[['x1','y1','z1']].max(axis=1)
df3['Max2'] = df3[['x2','y2','z2']].max(axis=1)
df3['Max3'] = df3[['x3','y3','z3']].max(axis=1)
df3['Max4'] = df3[['x4','y4','z4']].max(axis=1)
df3['Max'] = df3[features].max(axis=1)


# Min
df3['Min1'] = df3[['x1','y1','z1']].min(axis=1)
df3['Min2'] = df3[['x2','y2','z2']].min(axis=1)
df3['Min3'] = df3[['x3','y3','z3']].min(axis=1)
df3['Min4'] = df3[['x4','y4','z4']].min(axis=1)
df3['Min'] = df3[features].min(axis=1)


# Skewness
df3['Skew1'] = df3[['x1','y1','z1']].skew(axis=1)
df3['Skew2'] = df3[['x2','y2','z2']].skew(axis=1)
df3['Skew3'] = df3[['x3','y3','z3']].skew(axis=1)
df3['Skew4'] = df3[['x4','y4','z4']].skew(axis=1)
df3['Skew'] = df3[features].skew(axis=1)


# Kurtosis
df3['Kurt'] = df3[features].kurtosis(axis=1)


df3['class'] = 3



# Class 4 PreProcessing


df4 = df.loc[(df['class'] == 4)]
df4 = df4[features]
#df4 = df4.rolling(1000).mean().dropna()
#df4 = (df4- df4.min()) / (df4.max() - df4.min())
#df4 = (df4 - df4.mean())/ df4.std()
#df4 = pd.DataFrame(butter_lowpass_filter(df4, cutoff, fs, order),columns=[features])


# Class 4 Feature Extraction
# Mean
df4['Mean1'] = df4[['x1','y1','z1']].mean(axis=1)
df4['Mean2'] = df4[['x2','y2','z2']].mean(axis=1)
df4['Mean3'] = df4[['x3','y3','z3']].mean(axis=1)
df4['Mean4'] = df4[['x4','y4','z4']].mean(axis=1)
df4['Mean'] = df4[features].mean(axis=1)


# Norms
df4['Norm1'] = np.sqrt(np.square(df4[['x1','y1','z1']]).sum(axis=1))
df4['Norm2'] = np.sqrt(np.square(df4[['x2','y2','z2']]).sum(axis=1))
df4['Norm3'] = np.sqrt(np.square(df4[['x3','y3','z3']]).sum(axis=1))
df4['Norm4'] = np.sqrt(np.square(df4[['x4','y4','z4']]).sum(axis=1))
df4['Norm'] = np.sqrt(np.square(df4[features]).sum(axis=1))


# RMS
df4['RMS1'] = np.sqrt((1/3)*np.square(df4[['x1','y1','z1']]).sum(axis=1))
df4['RMS2'] = np.sqrt((1/3)*np.square(df4[['x2','y2','z2']]).sum(axis=1))
df4['RMS3'] = np.sqrt((1/3)*np.square(df4[['x3','y3','z3']]).sum(axis=1))
df4['RMS4'] = np.sqrt((1/3)*np.square(df4[['x4','y4','z4']]).sum(axis=1))
df4['RMS'] = np.sqrt((1/12)*np.square(df4[features]).sum(axis=1))


# Variance
df4['Var1'] = df4[['x1','y1','z1']].var(axis=1)
df4['Var2'] = df4[['x2','y2','z2']].var(axis=1)
df4['Var3'] = df4[['x3','y3','z3']].var(axis=1)
df4['Var4'] = df4[['x4','y4','z4']].var(axis=1)
df4['Var'] = df4[features].var(axis=1)


# Std
df4['Std1'] = df4[['x1','y1','z1']].std(axis=1)
df4['Std2'] = df4[['x2','y2','z2']].std(axis=1)
df4['Std3'] = df4[['x3','y3','z3']].std(axis=1)
df4['Std4'] = df4[['x4','y4','z4']].std(axis=1)
df4['Std'] = df4[features].std(axis=1)


# differences
df4[['DX1','DY1','DZ1','DX2','DY2','DZ2','DX3','DY3','DZ3','DX4','DY4','DZ4']] = df4[features].diff()
df4 = df4.dropna()


# Median
df4['Med1'] = df4[['x1','y1','z1']].median(axis=1)
df4['Med2'] = df4[['x2','y2','z2']].median(axis=1)
df4['Med3'] = df4[['x3','y3','z3']].median(axis=1)
df4['Med4'] = df4[['x4','y4','z4']].median(axis=1)
df4['Med'] = df4[features].median(axis=1)


# Max
df4['Max1'] = df4[['x1','y1','z1']].max(axis=1)
df4['Max2'] = df4[['x2','y2','z2']].max(axis=1)
df4['Max3'] = df4[['x3','y3','z3']].max(axis=1)
df4['Max4'] = df4[['x4','y4','z4']].max(axis=1)
df4['Max'] = df4[features].max(axis=1)


# Min
df4['Min1'] = df4[['x1','y1','z1']].min(axis=1)
df4['Min2'] = df4[['x2','y2','z2']].min(axis=1)
df4['Min3'] = df4[['x3','y3','z3']].min(axis=1)
df4['Min4'] = df4[['x4','y4','z4']].min(axis=1)
df4['Min'] = df4[features].min(axis=1)


# Skewness
df4['Skew1'] = df4[['x1','y1','z1']].skew(axis=1)
df4['Skew2'] = df4[['x2','y2','z2']].skew(axis=1)
df4['Skew3'] = df4[['x3','y3','z3']].skew(axis=1)
df4['Skew4'] = df4[['x4','y4','z4']].skew(axis=1)
df4['Skew'] = df4[features].skew(axis=1)


# Kurtosis
df4['Kurt'] = df4[features].kurtosis(axis=1)


df4['class'] = 4





# Class 5 PreProcessing


df5 = df.loc[(df['class'] == 5)]
df5 = df5[features]
#df5 = df5.rolling(1000).mean().dropna()
#df5 = (df5- df5.min()) / (df5.max() - df5.min())
#df5 = (df5 - df5.mean())/ df5.std()
#df5 = pd.DataFrame(butter_lowpass_filter(df5, cutoff, fs, order),columns=[features])


# Class 5 Feature Extraction
# Mean
df5['Mean1'] = df5[['x1','y1','z1']].mean(axis=1)
df5['Mean2'] = df5[['x2','y2','z2']].mean(axis=1)
df5['Mean3'] = df5[['x3','y3','z3']].mean(axis=1)
df5['Mean4'] = df5[['x4','y4','z4']].mean(axis=1)
df5['Mean'] = df5[features].mean(axis=1)


# Norms
df5['Norm1'] = np.sqrt(np.square(df5[['x1','y1','z1']]).sum(axis=1))
df5['Norm2'] = np.sqrt(np.square(df5[['x2','y2','z2']]).sum(axis=1))
df5['Norm3'] = np.sqrt(np.square(df5[['x3','y3','z3']]).sum(axis=1))
df5['Norm4'] = np.sqrt(np.square(df5[['x4','y4','z4']]).sum(axis=1))
df5['Norm'] = np.sqrt(np.square(df5[features]).sum(axis=1))


# RMS
df5['RMS1'] = np.sqrt((1/3)*np.square(df5[['x1','y1','z1']]).sum(axis=1))
df5['RMS2'] = np.sqrt((1/3)*np.square(df5[['x2','y2','z2']]).sum(axis=1))
df5['RMS3'] = np.sqrt((1/3)*np.square(df5[['x3','y3','z3']]).sum(axis=1))
df5['RMS4'] = np.sqrt((1/3)*np.square(df5[['x4','y4','z4']]).sum(axis=1))
df5['RMS'] = np.sqrt((1/12)*np.square(df5[features]).sum(axis=1))


# Variance
df5['Var1'] = df5[['x1','y1','z1']].var(axis=1)
df5['Var2'] = df5[['x2','y2','z2']].var(axis=1)
df5['Var3'] = df5[['x3','y3','z3']].var(axis=1)
df5['Var4'] = df5[['x4','y4','z4']].var(axis=1)
df5['Var'] = df5[features].var(axis=1)


# Std
df5['Std1'] = df5[['x1','y1','z1']].std(axis=1)
df5['Std2'] = df5[['x2','y2','z2']].std(axis=1)
df5['Std3'] = df5[['x3','y3','z3']].std(axis=1)
df5['Std4'] = df5[['x4','y4','z4']].std(axis=1)
df5['Std'] = df5[features].std(axis=1)


# differences
df5[['DX1','DY1','DZ1','DX2','DY2','DZ2','DX3','DY3','DZ3','DX4','DY4','DZ4']] = df5[features].diff()
df5 = df5.dropna()


# Median
df5['Med1'] = df5[['x1','y1','z1']].median(axis=1)
df5['Med2'] = df5[['x2','y2','z2']].median(axis=1)
df5['Med3'] = df5[['x3','y3','z3']].median(axis=1)
df5['Med4'] = df5[['x4','y4','z4']].median(axis=1)
df5['Med'] = df5[features].median(axis=1)


# Max
df5['Max1'] = df5[['x1','y1','z1']].max(axis=1)
df5['Max2'] = df5[['x2','y2','z2']].max(axis=1)
df5['Max3'] = df5[['x3','y3','z3']].max(axis=1)
df5['Max4'] = df5[['x4','y4','z4']].max(axis=1)
df5['Max'] = df5[features].max(axis=1)


# Min
df5['Min1'] = df5[['x1','y1','z1']].min(axis=1)
df5['Min2'] = df5[['x2','y2','z2']].min(axis=1)
df5['Min3'] = df5[['x3','y3','z3']].min(axis=1)
df5['Min4'] = df5[['x4','y4','z4']].min(axis=1)
df5['Min'] = df5[features].min(axis=1)


# Skewness
df5['Skew1'] = df5[['x1','y1','z1']].skew(axis=1)
df5['Skew2'] = df5[['x2','y2','z2']].skew(axis=1)
df5['Skew3'] = df5[['x3','y3','z3']].skew(axis=1)
df5['Skew4'] = df5[['x4','y4','z4']].skew(axis=1)
df5['Skew'] = df5[features].skew(axis=1)


# Kurtosis
df5['Kurt'] = df5[features].kurtosis(axis=1)


df5['class'] = 5





# Stack dataframes

df = pd.concat([df1,df2,df3,df4,df5])

print(len(df))

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

# Dimensionality Reduction
features = list(df.columns.values)
del features[len(features)-1]
x = df.loc[:, features].values
y = df.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

print(len(x),len(y))

#print(pd.DataFrame(data = x, columns = features).head())

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1','PC2','PC3','PC4','PC5'])

principalDf['class'] = df['class'].values
print(principalDf.head())

df_norm = principalDf

features = ['PC1','PC2','PC3','PC4','PC5']

testsize = 0.3

#features = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3','x4', 'y4', 'z4']

# Gaussian Fit
X = np.array(df_norm[features])
y = np.array(df_norm['class'].values.ravel())
seed = 7

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
df2 = df2.rolling(1000).mean().dropna()
df2 = (df2 - df2.min()) / (df2.max() - df2.min())
df2 = (df2 - df2.mean())/ df2.std()
df2 = pd.DataFrame(butter_lowpass_filter(df2, cutoff, fs, order),columns=[features])

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
df3 = df3.rolling(1000).mean().dropna()
df3 = (df3 - df3.min()) / (df3.max() - df3.min())
df3 = (df3 - df3.mean())/ df3.std()
df3 = pd.DataFrame(butter_lowpass_filter(df3, cutoff, fs, order),columns=[features])


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
df4 = df4.rolling(1000).mean().dropna()
df4 = (df4 - df4.min()) / (df4.max() - df4.min())
df4 = (df4 - df4.mean())/ df4.std()
df4 = pd.DataFrame(butter_lowpass_filter(df4, cutoff, fs, order),columns=[features])

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
df5 = df5.rolling(1000).mean().dropna()
df5 = (df5 - df5.min()) / (df5.max() - df5.min())
df5 = (df5 - df5.mean())/ df5.std()
df5 = pd.DataFrame(butter_lowpass_filter(df5, cutoff, fs, order),columns=[features])

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



train_X = np.zeros((8254,12,100))
train_Y = np.zeros((8254))
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
    
    
#train_X
    
    
    
   

    
    

"""

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