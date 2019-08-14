import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]])
y_data = np.array([[152.],
          [185.],
          [180.],
          [196.],
          [142.]])
epoch=1000
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=3))

model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(x_data, y_data, epochs=epoch)

y_predict = model.predict(np.array([[95., 100., 80]]))
print(y_predict)

epochs = np.arange(1, epoch+1)
plt.plot(epochs, history.history['loss'], label='Training loss')
#plt.plot(epochs, history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()