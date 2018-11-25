import tensorflow as tf 
import numpy as np 

#n_step = 
n_input = 3
n_output = 1
n_hidden = 128

X1 = tf.constant([[1,2,3],[2,3,4],[5,6,7]])
X2 = tf.constant([[1,2,3],[2,3,4],[5,6,7]])
#X3 = [[1,2,3],[2,3,4],[5,6,7]]
Y1 = tf.add(X1,Y1)

#manul implementation
X_batch = tf.placeholder(tf.float32,shape=[None,3])
Y_batch = tf.placeholder(tf.float32,shape=[None,3])

#notes: there is only one biase
W_xh = tf.Variable(tf.truncated_normal(shape=[n_input,n_hidden]))
#b1 = tf.Variable(tf.constant(shape=[n_hidden]))
W_hh = tf.Variable(tf.truncated_normal(shape=[n_hidden,n_hidden]))
b = tf.Variable(tf.constant(shape=[n_output]))

#the dimension of hidden1 is 
h1 = tf.nn.tanh(tf.matmul(X_batch[][0],W_xh) +b)
h2 = tf.nn.tanh(tf.nn.tanh(tf.matmul(X_batch[][1],W_xh) + tf.matmul(h1, W_hh) +b))
#h3 = tf.nn.tanh(tf.nn.tanh(tf.matmul(X_batch[][2],W_xh) + tf.matmul(h2, W_hh) +b))

#cell 
cell = tf.contrib.
init_state = cell.zero
output, state = tf.contrib.dynami(cell,)

with tf.Session() as sess:
    