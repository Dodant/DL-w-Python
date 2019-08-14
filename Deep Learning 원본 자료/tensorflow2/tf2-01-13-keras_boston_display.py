# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:50:23 2019

@author: user
"""

import tensorflow as tf

(X_train, y_train),(X_test,y_test) = tf.keras.datasets.boston_housing.load_data()
print("Training Data :","X_train",X_train.shape,"y_train", y_train.shape,", Test Data :","X_test", X_test.shape,"y_test", y_test.shape)

for i in range(5):
    print(X_train[i,:], y_train[i])
    
print()
print('crim :',X_train[0,0], 'zn :',X_train[0,1], 'indus :',X_train[0,2],'chas :',X_train[0,3])
print('nox :',X_train[0,4], 'rm :',X_train[0,5], 'age :',X_train[0,6],'dis :',X_train[0,7])
print('rad :',X_train[0,8], 'tax :',X_train[0,9], 'ptratio :',X_train[0,10],'black :',X_train[0,11],'lstat :',X_train[0,12])
print('medv :',y_train[0])

