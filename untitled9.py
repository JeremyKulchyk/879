#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 22:34:40 2018

@author: JeremyKulchyk



Tables/IMAGES:
    
    
INSERT ALL OF
    - All 12 signals plotted in same graph
    - Raw X1 Signal
    - Rolling mean of 100, 500, 1000, 1500 of X1 Signal
    - HP butterowrth of X1
    - X1 after norm/std
    - CNN Model
    - NB: Best results and preprocessing steps
    - DT: Best results and preprocessing steps
    - KNN: Best results and preprocessing steps
    - E2: Best results and preprocessing steps
    - RF: Best results and preprocessing steps
    - CNN: Show results with and without normalization
    - Comparison between all models
    - 

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

df1 = df.loc[(df['class'] == 1)]
df2 = df.loc[(df['class'] == 2)]
df3 = df.loc[(df['class'] == 3)]
df4 = df.loc[(df['class'] == 4)]
df5 = df.loc[(df['class'] == 5)]

R = 1000
df1_ = df1['x3']#.rolling(R).mean()
df2_ = df2['x3']#.rolling(R).mean()
df3_ = df3['x3']#.rolling(R).mean()
df4_ = df4['x3']#.rolling(R).mean()
df5_ = df5['x3']#.rolling(R).mean()


import matplotlib.patches as mpatches

C1 = mpatches.Patch(color='blue', label='Sitting')
C2 = mpatches.Patch(color='green', label='Sitting Down')
C3 = mpatches.Patch(color='orange', label='Standing')
C4 = mpatches.Patch(color='red', label='Standing Up')
C5 = mpatches.Patch(color='purple', label='Walking')

"""
plt.plot(df1_)
plt.plot(df2_)
plt.plot(df3_)
plt.plot(df4_)
plt.plot(df5_)
plt.legend(handles=[C1,C2,C3,C4,C5])
plt.title("X3 Component - Rolling Mean (1500)")
plt.xlabel('Time Steps')
plt.ylabel('Acceleration')
#plt.savefig("X3_Roll_1500")
"""

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
plt.title("Highpass Filter Frequency Response of X3")
plt.xlabel('Frequency')
plt.ylabel('Gain')
plt.grid()

# Filter the data, and plot both the original and filtered signals.
df_norm_F = pd.DataFrame(butter_lowpass_filter(df_norm, cutoff, fs, order),columns=[features])

plt.subplot(2, 1, 2)
plt.plot(df_norm['x3'].rolling(R).mean(), 'b-', label='data')
plt.plot(df_norm_F['x3'].rolling(R).mean(), 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.savefig("X3_HPF")
plt.show()


#df_denoiseGauss = df_norm.rolling(window=10000, win_type='gaussian', center=True).mean(std=0.5)


# Remove Noise
df_norm = df_norm_F.rolling(1000).mean()
#plt.plot(df_norm['x1'])

# Unity Normalization
df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
#plt.plot(df_norm['x1'])'


# Standarize
df_norm = (df_norm - df_norm.mean())/ df_norm.std()

plt.plot(df_norm['x3'])
plt.legend()
plt.title("X3 Component - Normalized & Standardized")
plt.xlabel('Time Steps')
plt.ylabel('Values')
#plt.savefig("X3_NormStd")






