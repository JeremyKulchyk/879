#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:10:05 2018


3 different ensembles:
    1. Ensemble of 4 weak CNN's with each sensor
    2. Ensemble of 3 weak CNN's with each component
    3. Baseline model of all 12 components

Monday:
    Test performance of each.
    Generate charts/plots for each.
    
Tuesday:
    Write report

Wednesday:
    Show Ali
    
    
Try 3 signals at a time - 220 possible combinations
    Compute Avg correlation of
    12 choose 3


Loop through all 220 combinations of 3 signals (3x3 conv used to encapture correlation of all signals)
for each, record:
    avg correlation amongst each set of 3
    training loss, acc
    validation loss, acc
    test error, acc, all performance measures

Compare performance with correlation
Possibly deduce relationship, wrt sensor components

Deduced what 3 components matter most,
Best 3 use x and y of foot (3), and any component of shoulder (4)
    


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
        
print(df.head())

# 4. Noise Reduction
# Changed -14420-11-2011 04:50:23.713 at (122078,Z4) to previous value (-142)


# Segmentation - window size: 100, overlap: 80%
Window = 100
overlap = 0.80
step = int(Window - Window*0.80)

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
from keras.models import model_from_json


train_X = np.zeros((8254,12,100))
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
"""
for i in range(0,len(df2Seg)):
    train_X = np.append(train_X,df2Seg[i])
    train_Y = np.append(train_Y,2)
for i in range(0,len(df3Seg)):
    train_X = np.append(train_X,df3Seg[i])
    train_Y = np.append(train_Y,3)
for i in range(0,len(df4Seg)):
    train_X = np.append(train_X,df4Seg[i])
    train_Y = np.append(train_Y,4)
for i in range(0,len(df5Seg)):
    train_X = np.append(train_X,df5Seg[i])
    train_Y = np.append(train_Y,5)
"""


"""
train_X = []
for i in range(0,int(0.8*len(df1Seg))):
    train_X.append(df1Seg[i])
for i in range(0,int(0.8*len(df2Seg))):
    train_X.append(df2Seg[i])
for i in range(0,int(0.8*len(df3Seg))):
    train_X.append(df3Seg[i])
for i in range(0,int(0.8*len(df4Seg))):
    train_X.append(df4Seg[i])
for i in range(0,int(0.8*len(df5Seg))):
    train_X.append(df5Seg[i])

train_Y = []
for i in range(0,int(0.8*len(df1Seg))):
    train_Y.append(1)
for i in range(0,int(0.8*len(df2Seg))):
    train_Y.append(2)
for i in range(0,int(0.8*len(df3Seg))):
    train_Y.append(3)
for i in range(0,int(0.8*len(df4Seg))):
    train_Y.append(4)
for i in range(0,int(0.8*len(df5Seg))):
    train_Y.append(5)
print(len(train_Y))

test_X = []
for i in range(int(0.8*len(df1Seg)),len(df1Seg)):
    test_X.append(df1Seg[i])
for i in range(int(0.8*len(df2Seg)),len(df2Seg)):
    test_X.append(df2Seg[i])
for i in range(int(0.8*len(df3Seg)),len(df3Seg)):
    test_X.append(df3Seg[i])
for i in range(int(0.8*len(df4Seg)),len(df4Seg)):
    test_X.append(df1Seg[i])
for i in range(int(0.8*len(df5Seg)),len(df5Seg)):
    test_X.append(df5Seg[i])

test_Y = []
for i in range(int(0.8*len(df1Seg)),len(df1Seg)):
    test_Y.append(1)
for i in range(int(0.8*len(df2Seg)),len(df2Seg)):
    test_Y.append(2)
for i in range(int(0.8*len(df3Seg)),len(df3Seg)):
    test_Y.append(3)
for i in range(int(0.8*len(df4Seg)),len(df4Seg)):
    test_Y.append(4)
for i in range(int(0.8*len(df5Seg)),len(df5Seg)):
    test_Y.append(5)

print(len(train_X),len(train_Y),len(test_X),len(test_Y))
"""

print(train_X.shape)
train_X = train_X.reshape(-1, 12,100, 1)
print(train_X.shape)
print(train_Y.shape)
#print(train_X.shape)
"""
from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

print(train_X.shape)
train_X = train_X.reshape(-1, 28,28, 1)
print(train_X.shape)
"""
# define 10-fold cross validation test harness
# fix random seed for reproducibility
Dict = {}
batch_size = 64
epochs = 20
num_classes = 5
Count = 0
seed = 7
np.random.seed(seed)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=8254, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(train_X, train_Y):
    CNN1 = Sequential()
    CNN1.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(12,100,1),padding='same'))
    CNN1.add(LeakyReLU(alpha=0.1))
    CNN1.add(MaxPooling2D((2, 2),padding='same'))
    CNN1.add(Dropout(0.25))
    CNN1.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    CNN1.add(LeakyReLU(alpha=0.1))
    CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    CNN1.add(Dropout(0.25))
    CNN1.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    CNN1.add(LeakyReLU(alpha=0.1))                  
    CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    CNN1.add(Dropout(0.25))
    CNN1.add(Flatten())
    CNN1.add(Dense(128, activation='linear'))
    CNN1.add(LeakyReLU(alpha=0.1))                  
    CNN1.add(Dropout(0.25))
    CNN1.add(Dense(num_classes, activation='softmax'))
    CNN1.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    #print(CNN1.summary())
    CNN1.fit(train_X[train], train_Y[train], batch_size=batch_size,epochs=3,verbose=1)
    scores = CNN1.evaluate(train_X[test], train_Y[test], verbose=1)
    
    y_test = train_Y[test]
    y_pred = CNN1.predict(train_X[test])
    ypred = []
    for List in y_pred:
        M = max(List)
        i = np.where(List == M)[0][0]
        ypred.append(float(i))
    y_pred = ypred
    
    cm = confusion_matrix(y_test, y_pred)
    print("Accuaracy: ",accuracy_score(y_test, y_pred))
    print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
    print("Precision: ",precision_score(y_test, y_pred, average="macro"))
    print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
    print(cm)
    
    print("%s: %.2f%%" % (CNN1.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    Dict[Count] = {}
    Dict[Count]['cm'] = cm
    Dict[Count]['Accuracy'] = accuracy_score(y_test, y_pred)
    Dict[Count]['L1OAcc'] = scores[1]*100
    Dict[Count]['F1'] = accuracy_score(y_test, y_pred)
    Dict[Count]['Precision'] = accuracy_score(y_test, y_pred)
    Dict[Count]['Recall'] = accuracy_score(y_test, y_pred)
    """
    # serialize model to JSON
    model_json = CNN1.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        Name = "CNN" + str(Count) + ".h5"
        CNN1.save_weights(Name)
        print("Saved model to disk")
    """
    Count += 1
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print(Dict)

"""
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
"""

"""

from sklearn.model_selection import train_test_split
#train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)
train_X,valid_X,train_label,valid_label = cross_validation.train_test_split(train_X, train_Y, test_size=0.2, random_state=13)


CNN1 = Sequential()
CNN1.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(12,100,1),padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D((2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Flatten())
CNN1.add(Dense(128, activation='linear'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(Dropout(0.25))
CNN1.add(Dense(num_classes, activation='softmax'))

CNN1.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN1.summary())

CNN1 = CNN1.fit(train_X, train_label, batch_size=batch_size,epochs=20,verbose=1,validation_data=(valid_X, valid_label))
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
# Gaussian Fit
data = df_norm.values
target = df['class'].values
X = np.array(df_norm)
y = np.array(df['class'])
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=testsize)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#y_pred = gnb.fit(data, target).predict(data)
#print("# of mislabeled points/total %d points : %d" % (data.shape[0],(target != y_pred).sum()))
#accuracy = (data.shape[0]-(target != y_pred).sum())/data.shape[0]
print("NBG:")
print("Accuaracy: ",accuracy_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred, average="macro"))
print("Precision: ",precision_score(y_test, y_pred, average="macro"))
print("Recall: ",recall_score(y_test, y_pred, average="macro")) 
kfold = model_selection.KFold(n_splits=10)
model = clf
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print(" ")

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