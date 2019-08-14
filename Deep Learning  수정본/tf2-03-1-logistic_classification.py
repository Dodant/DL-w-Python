import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x_data = np.array([[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]])
y_data = np.array([[0],
          [0],
          [0],
          [1],
          [1],
          [1]])

plt.scatter(x_data[:3, 0], x_data[:3, 1],
            color='red', marker='o', label='class 0')
plt.scatter(x_data[3:, 0], x_data[3:, 1],
            color='blue', marker='x', label='class 1')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower right')
plt.show()

epoch=2000
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.summary()
history=model.fit(x_data, y_data, epochs=epoch)

score = model.evaluate(x_data, y_data, verbose=0)
print()
print('Test loss = ', score)

print("2,1", model.predict_classes(np.array([[2, 1]])))
print("6,5", model.predict_classes(np.array([[6, 5]])))

epochs = np.arange(1, epoch+1)
plt.plot(epochs, history.history['loss'], label='Training loss')
#plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.scatter(x_data[:3, 0], x_data[:3, 1],
            color='red', marker='o', label='0')
plt.scatter(x_data[3:, 0], x_data[3:, 1],
            color='blue', marker='x', label='1')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower right')

resolution=0.02
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:2])
x1_min, x1_max = x_data[:, 0].min() - 0.5, x_data[:, 0].max() + 0.5
x2_min, x2_max = x_data[:, 1].min() - 0.5, x_data[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

plt.show()
