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

features1 = ['x1', 'y1', 'z1']
features2 = ['x2', 'y2', 'z2']
features3 = ['x3', 'y3', 'z3']
features4 = ['x4', 'y4', 'z4']


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

# 4. Noise Reduction
# Changed -14420-11-2011 04:50:23.713 at (122078,Z4) to previous value (-142)


# Segmentation - window size: 100, overlap: 80%
Window = 100
overlap = 0.80
step = int(Window - Window*0.80)

features = features1
# Class 1 Segmentation
df1 = df.loc[(df['class'] == 1)]
df1 = df1[features]
df1 = (df1[features]- df1[features].min()) / (df1[features].max() - df1[features].min())
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
df2 = (df2[features]- df2[features].min()) / (df2[features].max() - df2[features].min())
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
df3 = (df3[features]- df3[features].min()) / (df3[features].max() - df3[features].min())
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
df4 = (df4[features]- df4[features].min()) / (df4[features].max() - df4[features].min())
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
df5 = (df5[features]- df5[features].min()) / (df5[features].max() - df5[features].min())
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

# CNN 1
# 12x100 Input size
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 5


train_X = np.zeros((8254,3,100))
train_Y = np.zeros((8254))
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i].transpose()
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i].transpose()
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i].transpose()
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i].transpose()
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i].transpose()
    train_Y[j] = 4
    j += 1


print(train_X.shape)
train_X = train_X.reshape(-1, 3,100, 1)
#print(train_X.shape)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
#train_X,valid_X,train_label,valid_label = cross_validation.train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


CNN1 = Sequential()
CNN1.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(3,100,1),padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D((2, 2),padding='same'))
CNN1.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Flatten())
CNN1.add(Dense(128, activation='linear'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(Dense(num_classes, activation='softmax'))

CNN1.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN1.summary())

CNN1 = CNN1.fit(train_X, train_label, batch_size=batch_size,epochs=5,verbose=1,validation_data=(valid_X, valid_label))



"""






CNN 2








"""

features = features2
# Class 1 Segmentation
df1 = df.loc[(df['class'] == 1)]
df1 = df1[features]
df1 = (df1[features]- df1[features].min()) / (df1[features].max() - df1[features].min())
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
df2 = (df2[features]- df2[features].min()) / (df2[features].max() - df2[features].min())
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
df3 = (df3[features]- df3[features].min()) / (df3[features].max() - df3[features].min())
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
df4 = (df4[features]- df4[features].min()) / (df4[features].max() - df4[features].min())
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
df5 = (df5[features]- df5[features].min()) / (df5[features].max() - df5[features].min())
Len = len(df5)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df5Seg = []
i = 0
while i < Len:
    df5Seg.append(df5.iloc[i:i+Window].values)
    i += step

batch_size = 64
epochs = 20
num_classes = 5

train_X = np.zeros((8254,3,100))
train_Y = np.zeros((8254))
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i].transpose()
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i].transpose()
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i].transpose()
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i].transpose()
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i].transpose()
    train_Y[j] = 4
    j += 1


print(train_X.shape)
train_X = train_X.reshape(-1, 3,100, 1)
#print(train_X.shape)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
#train_X,valid_X,train_label,valid_label = cross_validation.train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


CNN2 = Sequential()
CNN2.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(3,100,1),padding='same'))
CNN2.add(LeakyReLU(alpha=0.1))
CNN2.add(MaxPooling2D((2, 2),padding='same'))
CNN2.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN2.add(LeakyReLU(alpha=0.1))
CNN2.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN2.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN2.add(LeakyReLU(alpha=0.1))                  
CNN2.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN2.add(Flatten())
CNN2.add(Dense(128, activation='linear'))
CNN2.add(LeakyReLU(alpha=0.1))                  
CNN2.add(Dense(num_classes, activation='softmax'))

CNN2.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN2.summary())

CNN2 = CNN2.fit(train_X, train_label, batch_size=batch_size,epochs=5,verbose=1,validation_data=(valid_X, valid_label))


"""




 CNN 3







"""

"""






CNN 2








"""

features = features3
# Class 1 Segmentation
df1 = df.loc[(df['class'] == 1)]
df1 = df1[features]
df1 = (df1[features]- df1[features].min()) / (df1[features].max() - df1[features].min())
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
df2 = (df2[features]- df2[features].min()) / (df2[features].max() - df2[features].min())
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
df3 = (df3[features]- df3[features].min()) / (df3[features].max() - df3[features].min())
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
df4 = (df4[features]- df4[features].min()) / (df4[features].max() - df4[features].min())
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
df5 = (df5[features]- df5[features].min()) / (df5[features].max() - df5[features].min())
Len = len(df5)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df5Seg = []
i = 0
while i < Len:
    df5Seg.append(df5.iloc[i:i+Window].values)
    i += step

batch_size = 64
epochs = 20
num_classes = 5

train_X = np.zeros((8254,3,100))
train_Y = np.zeros((8254))
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i].transpose()
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i].transpose()
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i].transpose()
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i].transpose()
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i].transpose()
    train_Y[j] = 4
    j += 1


print(train_X.shape)
train_X = train_X.reshape(-1, 3,100, 1)
#print(train_X.shape)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
#train_X,valid_X,train_label,valid_label = cross_validation.train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


CNN3 = Sequential()
CNN3.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(3,100,1),padding='same'))
CNN3.add(LeakyReLU(alpha=0.1))
CNN3.add(MaxPooling2D((2, 2),padding='same'))
CNN3.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN3.add(LeakyReLU(alpha=0.1))
CNN3.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN3.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN3.add(LeakyReLU(alpha=0.1))                  
CNN3.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN3.add(Flatten())
CNN3.add(Dense(128, activation='linear'))
CNN3.add(LeakyReLU(alpha=0.1))                  
CNN3.add(Dense(num_classes, activation='softmax'))

CNN3.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN3.summary())

CNN3 = CNN3.fit(train_X, train_label, batch_size=batch_size,epochs=5,verbose=1,validation_data=(valid_X, valid_label))



""""




CNN4






"""



features = features4
# Class 1 Segmentation
df1 = df.loc[(df['class'] == 1)]
df1 = df1[features]
df1 = (df1[features]- df1[features].min()) / (df1[features].max() - df1[features].min())
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
df2 = (df2[features]- df2[features].min()) / (df2[features].max() - df2[features].min())
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
df3 = (df3[features]- df3[features].min()) / (df3[features].max() - df3[features].min())
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
df4 = (df4[features]- df4[features].min()) / (df4[features].max() - df4[features].min())
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
df5 = (df5[features]- df5[features].min()) / (df5[features].max() - df5[features].min())
Len = len(df5)
while Len%step != 0:
    Len = Len -1
Len = Len - Window
df5Seg = []
i = 0
while i < Len:
    df5Seg.append(df5.iloc[i:i+Window].values)
    i += step

batch_size = 64
epochs = 20
num_classes = 5

train_X = np.zeros((8254,3,100))
train_Y = np.zeros((8254))
j = 0
for i in range(0,len(df1Seg)):
    train_X[j] = df1Seg[i].transpose()
    train_Y[j] = 0
    j += 1
for i in range(0,len(df2Seg)):
    train_X[j] = df2Seg[i].transpose()
    train_Y[j] = 1
    j += 1
for i in range(0,len(df3Seg)):
    train_X[j] = df3Seg[i].transpose()
    train_Y[j] = 2
    j += 1
for i in range(0,len(df4Seg)):
    train_X[j] = df4Seg[i].transpose()
    train_Y[j] = 3
    j += 1
for i in range(0,len(df5Seg)):
    train_X[j] = df5Seg[i].transpose()
    train_Y[j] = 4
    j += 1


print(train_X.shape)
train_X = train_X.reshape(-1, 3,100, 1)
#print(train_X.shape)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
#train_X,valid_X,train_label,valid_label = cross_validation.train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


CNN4 = Sequential()
CNN4.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(3,100,1),padding='same'))
CNN4.add(LeakyReLU(alpha=0.1))
CNN4.add(MaxPooling2D((2, 2),padding='same'))
CNN4.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN4.add(LeakyReLU(alpha=0.1))
CNN4.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN4.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN4.add(LeakyReLU(alpha=0.1))                  
CNN4.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN4.add(Flatten())
CNN4.add(Dense(128, activation='linear'))
CNN4.add(LeakyReLU(alpha=0.1))                  
CNN4.add(Dense(num_classes, activation='softmax'))

CNN4.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN4.summary())

CNN4 = CNN4.fit(train_X, train_label, batch_size=batch_size,epochs=5,verbose=1,validation_data=(valid_X, valid_label))





