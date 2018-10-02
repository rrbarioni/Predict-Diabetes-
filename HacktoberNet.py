#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:48:06 2018

@author: srinivas
"""

import numpy as np # moathematical operations
import matplotlib.pyplot as plt #plot charts
import pandas as pd # import and manage data sets

#import datasets
datasets=pd.read_csv('diabetes.csv')# dataset is a variable that stores data
X=datasets.iloc[: , 0:8].values#X contains elements from column 0 to 7
y=datasets.iloc[: , 8].values  # y contain elements of column 8

#dividing the date
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#implementing ANN

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Concatenate

input_shape = X_test[0].shape
model_input = Input(shape=input_shape)

def branch_block(x, n_branches):
    branches = []
    for i in range(n_branches):
        branch = Dense(input_shape[0], activation='relu')(x)
        branch = Dense(input_shape[0] // 2, activation='relu')(branch)
        branch = Dense(input_shape[0] // 4, activation='relu')(branch)
        branch = Dropout(0.1)(branch)
        
        branches += [branch]
        
    concat_branches = Concatenate()([x] + branches)
    
    return concat_branches

main_branch = branch_block(model_input, 2)
main_branch = branch_block(main_branch, 10)
main_branch = branch_block(main_branch, 2)
main_branch = branch_block(main_branch, 1)

output = Dense(1, activation='sigmoid')(main_branch)

model = Model(inputs=model_input, outputs=output)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=100)

y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)
test_acc_hacktobernet = np.count_nonzero(np.equal(y_pred[:,0], y_test) == 1) / len(y_pred[:,0])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
