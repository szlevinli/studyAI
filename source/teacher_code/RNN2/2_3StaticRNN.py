# load library
import os
import numpy as np
import tensorflow as tf

n_steps = 2
n_inputs = 3
n_units = 5
batch_size = 4

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
print("the shape of x is",x.shape)
x_seq = tf.unstack(x, num=n_steps, axis=1)
#the type of x_seq is list
print("the type of x_seq is",type(x_seq))

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_units)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
outputs, states = tf.contrib.rnn.static_rnn(cell, x_seq, initial_state=init_state)

init = tf.global_variables_initializer()


#x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
#x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

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

# h0_val, h1_val = sess.run([h0, h1], feed_dict={x0: x0_batch, x1: x1_batch})


print("the value of outputs_val[0] are")
print(outputs_val[0])


#print("the value of outputs_val are")
#print(outputs_val)