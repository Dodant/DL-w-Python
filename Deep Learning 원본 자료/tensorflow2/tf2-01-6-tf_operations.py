# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 19:26:21 2017

@author: user
"""
# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np

#%%
print("********** rank 1 - Numpy ****************")

t1 = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t1)
print(t1.ndim) # rank
print(t1.shape) # shape
print(t1[0], t1[1], t1[-1])
print(t1[2:5], t1[4:-1])
print(t1[:2], t1[3:])
print()

#%%
print("********** rank 2 - Numpy ****************")
t2 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t2)
print('rank = ', t2.ndim) # rank
print('shape = ', t2.shape) # shape
print()

#%%
print("********** rank 1 - Tensor ****************")
t1_1 = tf.constant([1,2,3,4])
print(t1_1)
print('rank = ', tf.rank(t1_1).numpy())
print('shape = ', t1_1.shape)
print()

#%%
print("********** rank 2 - Tensor ****************")
t2_1 = tf.constant([[1,2], [3,4]])
print(t2_1)
print('rank = ', tf.rank(t2_1).numpy())
print('shape = ', t2_1.shape)
print()

#%%
print("********** rank 3 - Tensor ****************")
t3 = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(t3)
print('rank = ', tf.rank(t3).numpy())
print('shape = ', t3.shape)
print()

#%%
print("********** rank 4 - Tensor ****************")
t4 = tf.constant([
    [
        [
            [1,2,3,4], 
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20], 
            [21,22,23,24]
        ]
    ]
])
print(t4)
print('rank = ', tf.rank(t4).numpy())
print('shape = ', t4.shape)
print()
        
#%%
print("********** matmul ****************")
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.],[2.]])
matmul = tf.matmul(matrix1, matrix2)
print(matrix1)
print(matrix2)
print(matmul)
print()

#%% # Broadcasting
print("********** multiply & Broadcasting ****************")
multiply1 = matrix1*matrix2
print(multiply1)
multiply2 = tf.multiply(matrix1, matrix2)
print(multiply2)
print()

#%%
print("********** Add ****************")
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
matadd2 = matrix1+matrix2
print(matrix2)
print()

print("********** Add & Broadcasting ****************")
# Broadcasting
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
matadd1 = matrix1+matrix2
print(matadd1)
print()


#%%
print("********** range ****************")
start=3
limit=18
steps=3
print(tf.range(start, limit, steps)) # [3, 6, 9, 12, 15]

start=3
limit=1
steps=-0.5
print(tf.range(start, limit, steps)) # [3, 2.5, 2, 1.5]

limit=5
print(tf.range(limit)) # [0, 1, 2, 3, 4]
print()

#%%

print("********** random number generation ****************")
#%%

print(tf.random.normal([3]).numpy())
print(tf.random.uniform([2]))
print(tf.random.uniform([2, 3]))
print()

#%%
print("********** reduce mean & axis ****************")

x = [[1., 2.],
     [3., 4.]]

print(tf.reduce_mean(x))
print(tf.reduce_mean(x, axis=0)) # axis=0으로 reduce mean
print(tf.reduce_mean(x, axis=1)) # axis=1로 reduce mean
print()

#%%
print("********** Reshape ****************")
# reshape

t = tf.constant([[[0, 1, 2],
               [3, 4, 5]], 
              [[6, 7, 8],
               [9, 10, 11]]])

print(tf.shape(t))

print(tf.reshape(t, shape=[-1]))
print(tf.reshape(t, shape=[-1, 2]))
print(tf.reshape(t, shape=[-1, 3]))
print(tf.reshape(t, shape=[-1, 1, 3]))

print(tf.squeeze([[0], [1], [2]]))
print(tf.expand_dims([0, 1, 2], 1))
print()

#%%
print("********** One hot ****************")

print(tf.one_hot([[0], [1], [2], [0]], depth=3))
print()

#%%
print("********** Casting ****************")

print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32))
print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32))
print()

#%%
print("********** Stack ****************")

x = [1, 4]
y = [2, 5]
z = [3, 6]

print(tf.stack([x, y, z]))
print()

#%%
print("********** Ones and Zeros like ****************")

x = [[0, 1, 2],
     [2, 1, 0]]

print(tf.ones_like(x)) # 1로 채운
print(tf.zeros_like(x)) # 0으로 채운
print()

#%%
print("********** Zips ****************")

for x, y in zip([1, 2, 3], [4, 5, 6]): # x와 y에 담아서 처리
  print(x, y)






        