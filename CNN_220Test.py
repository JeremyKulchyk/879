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
    # 5 = walking
df = df.replace('sitting', 1)
df = df.replace('standing', 2)
df = df.replace('sittingdown', 3)
df = df.replace('standingup', 4)
df = df.replace('walking', 5)
        
dfCor = df[features].rolling(1000).mean()
"""
Dict = {}
for C in features:
    Dict[C] = []
    for D in features:
        if C == D:
            continue
        Cor = df[C].corr(df[D]).round(2)
        Dict[C].append(abs(Cor))
        if C == 'x2' or C == 'y2' or C == 'z2':
            print(C, D, Cor)
            
"""        
""" {'x2': [0.34, 0.96, 0.98], 'y2': [0.31, 0.91, 0.98], 'z2': [0.53, 0.91, 0.96]}
             y4    z2     y2           x1    z2     x2            y4   y2     x2

"""
features_12 = [['x1','y1', 'z1']]
for a in features:
    for b in features:
        for c in features:
            if a == b or a == c or b == c:
                continue
            List = [a,b,c]
            List1 = [a,c,b]
            List2 = [b,c,a]
            List3 = [b,a,c]
            List4 = [c,a,b]
            List5 = [c,b,a]
            if List not in features_12:
                if List1 not in features_12 and List2 not in features_12 and List3 not in features_12 and List4 not in features_12 and List5 not in features_12:
                    features_12.append(List)

#print(features_12)
print(len(features_12))

features_1 = [['x2','y4', 'z2'],['x1','y1', 'z1']]
features_1 = features_12

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
"""
df = df.replace('sitting', 1)
df = df.replace('standing', 2)
df = df.replace('sittingdown', 3)
df = df.replace('standingup', 4)
df = df.replace('walking', 5)
        
print(df.head())
"""
# 4. Noise Reduction
# Changed -14420-11-2011 04:50:23.713 at (122078,Z4) to previous value (-142)


# Segmentation - window size: 100, overlap: 80%
Window = 100
overlap = 0.80
step = int(Window - Window*0.80)

ModelDict = {}
Count = 0

for feat in features_1:
    features = feat
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
    """   
    print(len(df1),len(df1Seg))
    print(len(df2),len(df2Seg))
    print(len(df3),len(df3Seg))
    print(len(df4),len(df4Seg))
    print(len(df5),len(df5Seg))
    
    print(len(df1Seg) + len(df2Seg) + len(df3Seg) + len(df4Seg) + len(df5Seg))
    """
    # CNN 1
    # 12x100 Input size
    import keras
    from keras.models import Sequential,Input,Model
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import LeakyReLU
    
    batch_size = 64
    epochs = 5
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
    
    
    #print(train_X.shape)
    train_X = train_X.reshape(-1, 3,100, 1)
    #print(train_X.shape)

    from sklearn.model_selection import train_test_split
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.3, random_state=13)
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
    #print(CNN1.summary())
    
    CNN1.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(valid_X, valid_label))
    #scores = CNN1.evaluate(valid_X, valid_label, verbose=0)

    y_test = valid_label
    y_pred = CNN1.predict(valid_X)
    ypred = []
    for List in y_pred:
        M = max(List)
        i = np.where(List == M)[0][0]
        ypred.append(float(i))
    y_pred = ypred
    
    CM = confusion_matrix(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro")
    Precision = precision_score(y_test, y_pred, average="macro")
    Recall = recall_score(y_test, y_pred, average="macro")
    
    #print("Accuaracy: ",accuracy_score(y_test, y_pred))
    #print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
    #print("Precision: ",precision_score(y_test, y_pred, average="macro"))
    #print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
    #print(CM)
    
    
    Sum = 0
    Sum = Sum + abs(dfCor[feat[0]].corr(dfCor[feat[1]]).round(2))
    Sum = Sum + abs(dfCor[feat[0]].corr(dfCor[feat[2]]).round(2))
    Sum = Sum + abs(dfCor[feat[1]].corr(dfCor[feat[2]]).round(2))
    AvgCor = Sum/3
    
    # serialize model to JSON
    model_json = CNN1.to_json()
    Name = ""
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        Name = "CNN" + str(Count) + ".h5"
        CNN1.save_weights(Name)
        print("Saved model to disk: " + Name)
    Count += 1
    ModelDict[Name] = {"Components": feat,
                      "CM":CM,
                      "AvgCor":AvgCor,
                      "Accuraccy":Accuracy,
                      "F1": F1,
                      "Precision":Precision,
                      "Recall":Recall}
    print(Count)
 
print(ModelDict)
# later...
 
"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("CNN1.h5")
print("Loaded model from disk")
print(loaded_model)
"""

"""
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

CNN1 = CNN1.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
"""









