# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:38:39 2019

@author: user
"""
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import utils

import matplotlib.pyplot as plt

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

timesteps = seq_length = 7
data_dim = 5

# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# very important. It does not work without it.
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

model = Sequential()
model.add(GRU(1, input_shape=(seq_length, data_dim), 
               return_sequences=False))

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
model.summary()

print(trainX.shape, trainY.shape)
history=model.fit(trainX, trainY, epochs=200, verbose=2)

# make predictions
testPredict = model.predict(testX)

# print(testPredict)
plt.plot(testY, label='Target data')
plt.plot(testPredict, label='Predicted  data')
plt.legend()
plt.show()

history_dict = history.history
loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, '-', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
