import tensorflow as tf 


def nn_add_one_layer(input, insize, outsize,activation_function=None):
    W = tf.Varaible(tf.truncated_normal((insize,outsize)),dtype=tf.float32)
    b = tf.Variable(tf.truncated_normal((1,outsize)),,dtype=tf.float32)
    if activation_function is None:
       return tf.nn.relu(tf.matmul(input,W)+b)
    else:
       return activation_function(tf.matmul(input,W)+b)
