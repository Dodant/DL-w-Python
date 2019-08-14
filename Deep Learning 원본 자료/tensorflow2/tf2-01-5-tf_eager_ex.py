import tensorflow as tf

print(tf.executing_eagerly())

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m)) 
print("m:",m.numpy())

a = tf.constant([[1,2],[3,4]])
print(a) 
b = tf.add(a, 1)
print(b) 
print()

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('3과 5의 배수')
    elif int(num % 3) == 0:
      print('3의 배수')
    elif int(num % 5) == 0:
      print('5의 배수')
    else:
      print(num.numpy())
    counter += 1

fizzbuzz(15)

print()
print('Computing gradients')
W = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = W * W

grad = tape.gradient(loss, W)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)