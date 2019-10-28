# load library
import os
import numpy as np
import tensorflow as tf

n_steps = 2
n_inputs = 3
n_units = 5
batch_size = 4

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_units)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

init = tf.global_variables_initializer()

x_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    sess.run(init)
    outputs_val = sess.run(outputs,feed_dict={x: x_batch})

print(outputs_val)