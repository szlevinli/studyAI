# load library
import os
import numpy as np
import tensorflow as tf

# static unroll RNN
#tf.reset_default_graph()

n_inputs = 3
n_units = 5
batch_size = 4

x0 = tf.placeholder(tf.float32, [None, n_inputs])
x1 = tf.placeholder(tf.float32, [None, n_inputs])

#static_rnn
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_units)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = tf.contrib.rnn.static_rnn(cell, [x0, x1], initial_state=init_state)

#



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