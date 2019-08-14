# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:51 2019

@author: user
"""

import tensorflow as tf

target = ["negative sentiment", "positive sentiment"]

(X_train, y_train),(X_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=20000)
print(X_train.shape, X_test.shape)
print("- Training Data Length -")
print(len(X_train[0]),len(X_train[1]),len(X_train[2]),"...",len(X_train[24999]))
print()

word_index=tf.keras.datasets.imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = [reverse_word_index.get(i-3, ' ') for i in X_train[0]]

print('- decoded training data 1 - : target - %s' % target[y_train[0]])
for i in range(len(decoded_review)):
  if (i+1)%50==0:
    print("")
    
  print(decoded_review[i],end=' ')


print()
print()
decoded_review1 = [reverse_word_index.get(i-3, ' ') for i in X_train[1]]
print('- decoded training data 2 - : target - %s' % target[y_train[1]])
for i in range(len(decoded_review1)):
  if (i+1)%50==0:
    print("")
    
  print(decoded_review1[i],end=' ')