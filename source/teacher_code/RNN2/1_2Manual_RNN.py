# load library
import os
import numpy as np
import tensorflow as tf

# unroll RNN manually
n_inputs = 3
n_units = 5

x0 = tf.placeholder(tf.float32, [None, n_inputs])
x1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal([n_inputs, n_units], dtype=tf.float32), name="Wx")
Wh = tf.Variable(tf.random_normal([n_units, n_units], dtype=tf.float32), name="Wh")
b = tf.Variable(tf.zeros([n_units,], dtype=tf.float32), name="b")

h0 = tf.tanh(tf.matmul(x0, Wx) + b)
h1 = tf.tanh(tf.matmul(x1, Wx) + tf.matmul(h0, Wh) + b)

init = tf.global_variables_initializer()

x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    sess.run(init)
    h0_val, h1_val = sess.run([h0, h1], feed_dict={x0: x0_batch, x1: x1_batch})

print("the output of h0_val are")
print(h0_val)

print("the output of h1_val are")
print(h1_val)