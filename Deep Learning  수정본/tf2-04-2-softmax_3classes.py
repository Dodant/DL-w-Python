import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_scatter(X, y):
    # 마커와 컬러맵을 준비합니다
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # 클래스 샘플을 표시합니다
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap.colors[idx],
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    
    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그립니다
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict_classes(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.shape, Z.shape)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    plot_scatter(X, y)

 

m1 = [-1, 1]
cov1 = [[0.2, 0],[0, 0.2]]

m2 = [1, 1]
m3 = [0, -1]

x1 = np.random.multivariate_normal(m1, cov1, 100)
x2 = np.random.multivariate_normal(m2, cov1, 100)
x3 = np.random.multivariate_normal(m3, cov1, 100)
x_data = np.concatenate((x1, x2), axis=0)
x_data = np.concatenate((x_data, x3), axis=0)

y1 = np.ones((100, ), dtype='i')-1
y2 = 2*np.ones((100, ), dtype='i')-1
y3 = 3*np.ones((100, ), dtype='i')-1
y = np.concatenate((y1, y2), axis=0)
y = np.concatenate((y, y3), axis=0)

y_data = tf.keras.utils.to_categorical(y)

plot_scatter(x_data, y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

nb_classes = 3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=(2,)))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=500)

score = model.evaluate(x_data, y_data, verbose=0)
print()
print('Test loss = ', score[0], 'Test accuracy = ', score[1])
#print(model.predict_classes(x_data))

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


plot_decision_regions(x_data, y, model)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


