# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:11:29 2019

@author: user
"""

import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train),(X_test,y_test) = tf.keras.datasets.cifar100.load_data()
print("Training Data :","X_train",X_train.shape,"y_train", y_train.shape,", Test Data :","X_test", X_test.shape,"y_test", y_test.shape)


# Channel=0
for i in range(32):
    for j in range(32):
        print('%3d'%X_train[0,i,j,0], end=' ')
    print()


fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[i]
    ax[i].imshow(img, interpolation='nearest')
#    ax[i].imshow(img, interpolation='nearest')
    ax[i].set_title(' t: %d' % (y_train[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()