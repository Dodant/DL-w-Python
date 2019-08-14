import tensorflow as tf

# tensor variable 
t1 = tf.Variable(42) 
t2 = tf.Variable([1.,-1])
t3 = tf.Variable([[1.,-1.],[-1., 2]])
t4 = tf.Variable([[[0., 1., 2.],[3., 4., 5.]],[[6., 7., 8.],[9., 10., 11.]]]) 
print(t1)
print(t2)
print(t3)
print(t4)
#print(t1.shape, tf.rank(t1).numpy(), t2.shape, tf.rank(t2).numpy())

t1 = tf.add(t1, 1)
t2 = tf.add(t2, 1.0)
t3 = tf.add(t3, t2)
print(t1)
print(t2)
print(t3)



