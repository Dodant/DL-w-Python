# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:44:36 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

np.random.seed(777)

x_data = np.array([[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]])
y_data = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

# Evaluation our model using this test dataset
x_test = np.array([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])


model = Sequential()
model.add(Dense(16, input_dim=3))
model.add(Dense(3,activation='softmax'))

model.summary()

#sgd = SGD(lr=0.01)
#sgd = SGD(lr=1.0)
sgd = SGD(lr=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=500)

predictions = model.predict(x_test)
score = model.evaluate(x_test, y_test)

print('Prediction: ', [np.argmax(prediction) for prediction in predictions])
print('Accuracy: ', score[1])

history_dict = history.history
print(history_dict.keys())

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

# ‘bo’는 파란색 점을 의미합니다
plt.plot(epochs, loss, '-', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 그래프를 초기화합니다
acc = history.history['accuracy']

plt.plot(epochs, acc, '-', label='Training acc')
plt.title('Training  accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

