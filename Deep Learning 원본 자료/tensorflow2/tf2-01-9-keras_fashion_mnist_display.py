import tensorflow as tf
import matplotlib.pyplot as plt

label =['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
        'Sneaker', 'Bag', 'Ankle boot']
(train_x,train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

for i in range(28):
    for j in range(28):
        print('%3d'%train_x[0,i,j], end=' ')
    print()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = train_x[i]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#    ax[i].imshow(img, interpolation='nearest')
    ax[i].set_title('%s' % (label[train_y[i]]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#del train_x, train_y, test_x, test_y