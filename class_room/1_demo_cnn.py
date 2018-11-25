import tensorflow as tf
import numpy as np
import input_data

#! input data
mnist = input_data.read_data_sets('mnist_data', one_hot=True)
X_batch = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_batch = tf.placeholder(tf.float32, shape=[None, 10])

#! construct graph
#* conv shape: [ksize, ksize, in_size, out_size]
#* fully-connected shape: [in_size, out_size]
def get_weight(shape):
  return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32))

def get_bais(shape):
  return tf.constant(0.1, shape=shape, dtype=tf.float32)

#! add_layer(input, in_size, out_size, activation_function)
def conv2d(input, W, strides):
  # stides: the first on and the last must be 1
  # vaild: For 224*224, use 3*3 kernal size, the output is 222*222
  # SAME: For 224*224, use 3*3 kernal size, the output is 224*224
  return tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], pading='SAME')

def max_pool_2x2(input, strides, ksize):
  return tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME')

#! 核尺寸
kernal_size = 3

depth = 1
filter_num = 32
strides = 1

#! conv1 & pool
#* conv shape: [ksize, ksize, in_size, out_size]
conv1_weight = get_weight([kernal_size, kernal_size, depth, filter_num])
conv1_bais = get_bais([filter_num])
#* 输出特征图 shape: [None, 28, 28, 32]
conv1_conv = conv2d(X_batch, conv1_weight, strides=strides)
#* 池化层 shape: [None, 14, 14, 32]
conv1_pool = max_pool_2x2(conv1_conv, ksize=kernal_size, strides=2)