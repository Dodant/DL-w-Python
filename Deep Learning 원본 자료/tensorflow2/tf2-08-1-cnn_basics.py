# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:31:33 2018

@author: user
"""

#%matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("************* An Image - Gray *****************")
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]], 
                   [[7],[8],[9]]]], dtype=np.float32)
# print("imag:\n", image)
print("image.shape", image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
plt.show()


print("************* Conv2D - 1 filter - VALID *****************")
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d_img = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    
plt.show()

print("************* Conv2D - 1 filter - SAME *****************")
# print("imag:\n", image)
print("image.shape", image.shape)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d_img = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    
plt.show()

print("************* Conv2D - 3 filters *****************")
print("image.shape", image.shape)

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                      [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
print("weight.shape", weight.shape)
conv2d_img = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    
plt.show()


print("************* Max Pooling *****************")
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='VALID')
print(pool.shape)
#print(sess.run(pool))


print("************* Max Pooling - SAME *****************")
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
#print(sess.run(pool))

print("************* MNIST *****************")
(train_x,train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
img = train_x[0]
plt.imshow(img, cmap='gray')
plt.show()

print("************* MNIST - Conv2D *****************")
img = img.reshape(-1,28,28,1)
W1 = tf.Variable(tf.random.normal([3, 3, 1, 5], stddev=0.01))
conv2d_img = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
print(conv2d_img.shape)

conv2d_disp = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_disp):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
    
plt.show()   
   
print("************* MNIST - Max Pooling *****************") 
pool_img = tf.nn.max_pool(conv2d_img, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
print(pool_img.shape)

pool_disp = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_disp):
#    print(i, one_img.shape)
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
    
plt.show()
    
    

