# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:33:59 2019

@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=2, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t*t/3), size=None)
        y.append(r)
    return  x, 1.726*x -0.84 + np.array(y)

x, y = make_random_data() 

plt.plot(x, y, 'o')
plt.show()

epoch=100
x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
model.summary()

model.compile(optimizer='sgd', loss='mse')
history = model.fit(x_train, y_train, epochs=epoch, 
                    validation_split=0.3)
model.evaluate(x_test, y_test)

epochs = np.arange(1, epoch+1)
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

x_arr = np.arange(-2, 2, 0.1)
y_arr = model.predict(x_arr)

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', alpha=0.3)
plt.plot(x_arr, y_arr, '-r', lw=3)
plt.show()

model.save_weights('simple_weights.h5')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=1))
model.compile(optimizer='sgd', loss='mse')

model.load_weights('simple_weights.h5')
model.evaluate(x_test, y_test)

model.save('simple_model.h5')

model = tf.keras.models.load_model('simple_model.h5')
model.evaluate(x_test, y_test)