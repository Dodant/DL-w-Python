import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return(1. / (1. + np.exp(-x)))
    
def relu(x):
    return np.maximum(x, 0)
    
x_data = np.array([[0., 0.],
          [0., 1.],
          [1., 0.],
          [1., 1.]])
y_data = np.array([[0.],
          [1.],
          [1.],
          [0.]])

W1 = np.array([[1., 1.], [1.,1.]])
b1 = np.array([[0., -1.]])
W2 = np.array([[1., -2.]])
b2 = np.array([[0.]])

a1 = np.add(np.matmul(x_data, np.transpose(W1)), b1)
z1 = relu(a1)
a2 = np.add(np.matmul(z1, np.transpose(W2)), b2)
y = a2

y_pred = np.array(y>0.5, dtype=np.int32)
print(a1)
print(z1)
print(a2)
print(y)
print(y_pred)


plt.subplot(1,3,1)
plt.scatter(x_data[0, 0], x_data[0, 1], color='red', marker='o', label='0')
plt.scatter(x_data[3, 0], x_data[3, 1], color='red', marker='o', label='0')
plt.scatter(x_data[1, 0], x_data[1, 1], color='blue', marker='x', label='1')
plt.scatter(x_data[2, 0], x_data[2, 1], color='blue', marker='x', label='1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower center')
plt.title('x data')
plt.subplot(1,3,2)
plt.scatter(z1[0, 0], z1[0, 1], color='red', marker='o', label='0')
plt.scatter(z1[3, 0], z1[3, 1], color='red', marker='o', label='0')
plt.scatter(z1[1, 0], z1[1, 1], color='blue', marker='x', label='1')
plt.scatter(z1[2, 0], z1[2, 1], color='blue', marker='x', label='1')
plt.xlabel('z1')
plt.ylabel('z2')
#plt.legend(loc='lower center')
plt.title('z data after sigmoid')
plt.subplot(1,3,3)
plt.scatter(y[0], 0, color='red', marker='o', label='0')
plt.scatter(y[3], 0, color='red', marker='o', label='0')
plt.scatter(y[1], 0, color='blue', marker='x', label='1')
plt.scatter(y[2], 0, color='blue', marker='x', label='1')
plt.xlabel('y')
plt.yticks([0])
#plt.legend(loc='lower center')
plt.title('y data after sigmoid')


plt.show()
