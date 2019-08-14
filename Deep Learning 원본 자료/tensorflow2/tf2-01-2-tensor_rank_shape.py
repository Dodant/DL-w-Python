import tensorflow as tf
import numpy as np

# List
L0 = 3. # a rank 0 tensor; this is a scalar with shape []
L1 = [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
L2 = [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
L3 = [[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]] # a rank 3 tensor with shape [2, 1, 3]
L4 = [[[[1., 2],[3., 4.]], [[5., 6.],[7., 8]]], 
      [[[9., 10],[11., 12.]], [[13., 14.],[15., 16]]]]

print(type(L0))
print(type(L1))
print(type(L2))
print(type(L3))
print(type(L4))
print()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Numpy
N0 = np.array(L0)
N1 = np.array(L1)
N2 = np.array(L2)
N3 = np.array(L3)
N4 = np.array(L4)

print(type(N0))
print(type(N1))
print(type(N2))
print(type(N3))
print(type(N4))
print()

print('---------- Numpy dim -----------------')
print(N0.ndim)
print(N1.ndim)
print(N2.ndim)
print(N3.ndim)
print(N4.ndim)
print()

print('---------- Numpy shape ---------------')
print(N0.shape)
print(N1.shape)
print(N2.shape)
print(N3.shape)
print(N4.shape)
print()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Tensorflow
T0 = tf.constant(N0)
T1 = tf.constant(N1)
T2 = tf.constant(N2)
T3 = tf.constant(N3)
T4 = tf.constant(N4)

print("---------- Tensorflow -------------")
print(T0)
print(T1)
print(T2)
print(T3)
print(T4)
print()

print("---------- Tensorflow rank -------------")
print(tf.rank(T0), tf.rank(T0).numpy())
print(tf.rank(T1), tf.rank(T1).numpy())
print(tf.rank(T2), tf.rank(T2).numpy())
print(tf.rank(T3), tf.rank(T3).numpy())
print(tf.rank(T4), tf.rank(T4).numpy())
print()

print("---------- Tensorflow shape -------------")
print(T0.shape, tf.shape(T0), tf.shape(T0).numpy())
print(T1.shape, tf.shape(T1), tf.shape(T1).numpy())
print(T2.shape, tf.shape(T2), tf.shape(T2).numpy())
print(T3.shape, tf.shape(T3), tf.shape(T3).numpy())
print(T4.shape, tf.shape(T4), tf.shape(T4).numpy())
print()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

print("------------- Tensorflow -> Numpy -------------")
print(type(T2.numpy()), T2.numpy(), np.array(T2))
print(type(T4.numpy()), T4.numpy(), np.array(T4))
print((T2+T2).numpy())
print()

print("------------- Numpy -> Tensorflow -------------")
print(type(tf.constant(N2)), tf.constant(N2))
print(type(tf.constant(N4)), tf.constant(N4))
print(tf.add(T2, T2))
