# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:31:33 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import utils


# VGG16
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet')
vgg16.summary()


# VGG19
from tensorflow.keras.applications import VGG19

vgg19 = VGG19(weights='imagenet')
vgg19.summary()


# InceptionV3
from tensorflow.keras.applications import InceptionV3

inceptionv3 = InceptionV3(weights='imagenet')
inceptionv3.summary()

# DenseNet121
from tensorflow.keras.applications import DenseNet121

densenet121 = DenseNet121(weights='imagenet')
densenet121.summary()

# ResNet50
from tensorflow.keras.applications import ResNet50

resnet50 = ResNet50(weights='imagenet')
resnet50.summary()

# MobileNetV2
from tensorflow.keras.applications import MobileNetV2

mobilenetv2 = MobileNetV2(weights='imagenet')
mobilenetv2.summary()

# Xception
from tensorflow.keras.applications import Xception

xception = Xception(weights='imagenet')
xception.summary()

#batch_size = 100
#nb_classes = 10
#nb_epoch = 12
#
#img_rows, img_cols = 28, 28  # input image dimensions
#pool_size = (2, 2)  # size of pooling area for max pooling
#kernel_size = (3, 3)  # convolution kernel size
#
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
##X_train=np.reshape(X_train,(-1,X_train.shape[1],X_train.shape[2],3))
##X_test=np.reshape(X_test,(-1,X_test.shape[1],X_test.shape[2],3))
#
## convert class vectors to binary class matrices
#Y_train = utils.to_categorical(y_train, nb_classes)
#Y_test = utils.to_categorical(y_test, nb_classes)
#
#model = Sequential()
#
#model.add(Conv2D(32, kernel_size, padding='same', 
#                 input_shape=(X_train.shape[1],X_train.shape[2],3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(64, kernel_size, padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(128, kernel_size, padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=pool_size))
#model.add(Dropout(0.25))
#
#model.add(Flatten())
#model.add(Dense(600))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
#
#history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
#          verbose=2, validation_split=0.2)
#score = model.evaluate(X_test, Y_test, verbose=2)
#
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#
#history_dict = history.history
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(1, len(acc) + 1)
#
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#
#plt.show()
#
#plt.clf()   # 그래프를 초기화합니다
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#plt.plot(epochs, acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#
#plt.show()
    
    

