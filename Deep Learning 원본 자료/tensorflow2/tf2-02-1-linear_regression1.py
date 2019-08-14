import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2, 3])
y_train = np.array([1, 2, 3])

plt.plot(x_train, y_train, 'o')
plt.xticks(np.arange(0, 4.1, step=1))
plt.yticks(np.arange(0, 4.1, step=1))
plt.xlabel('x')
plt.ylabel('y')
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)
model.summary()

model.fit(x_train, y_train, epochs=100)

y_predict = model.predict(x_train)
print(y_predict)

x_arr = np.arange(-1, 4, 0.1)
y_arr = model.predict(x_arr)

plt.figure()
plt.plot(x_train, y_train, 'bo')
plt.plot(x_arr, y_arr, '-r', lw=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()