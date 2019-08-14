import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('data-01-test-score.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print("x_data", x_data)
print("y_data", y_data)
epoch=1000

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(input_dim=3, units=1))

model.compile(loss='mse', optimizer='rmsprop')
history=model.fit(x_data, y_data, epochs=epoch)

epochs = np.arange(1, epoch+1)
plt.plot(epochs, history.history['loss'], label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred = model.predict(x_data)
print(y_data, y_pred)
plt.plot(y_data, 'o', label='target')
plt.plot(y_pred, 'x', label='prediction')

plt.legend(fontsize='x-large')
plt.show()