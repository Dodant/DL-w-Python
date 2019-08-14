import tensorflow as tf

W = tf.Variable([2.])
b = tf.Variable([1.])
x = tf.constant([1., -1.])

print("W = ", W.numpy())
print("b = ", b.numpy())

h = tf.nn.relu(W*x + b)
print('x = ', x.numpy(), 'h = ', h.numpy())

