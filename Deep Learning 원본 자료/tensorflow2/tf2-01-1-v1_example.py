import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

W = tf.Variable([2.])
b = tf.Variable([1.])

x = tf.placeholder(tf.float32)

h = tf.nn.relu(W*x + b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("W = ", sess.run(W))
print("b = ", sess.run(b))
print("h = ", sess.run(h, feed_dict={x:1.}))
print("h = ", sess.run(h, feed_dict={x:-1.}))
