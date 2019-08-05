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

def get_bias(shape):
  return tf.constant(0.1, shape=shape, dtype=tf.float32)

#! add_layer(input, in_size, out_size, activation_function)
def conv2d(input, W, strides):
  # stides: the first on and the last must be 1
  # vaild: For 224*224, use 3*3 kernal size, the output is 222*222
  # SAME: For 224*224, use 3*3 kernal size, the output is 224*224
  return tf.nn.conv2d(input, W, strides=[1, strides, strides, 1], padding='SAME')

def max_pool_2x2(input, strides, ksize):
  return tf.nn.max_pool(input, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding='SAME')


step_num = 100
lr = 0.1
batch_size = 100

#! 内部变量定义
#* 核尺寸大小
kernal_size = 3
#* 原始图片的深度
depth = 1
#* 滤波器的数量 也就是通道的数量
filter_num = 32
#* 卷积层步长
conv_strides = 1
#* 池化层步长
pool_strides = 2

#! conv1 & pool
#* conv shape: [ksize, ksize, in_size, out_size]
conv1_weight = get_weight([kernal_size, kernal_size, depth, filter_num])
conv1_bias = get_bias([filter_num])
#* 输出结果
output1 = conv2d(X_batch, conv1_weight, strides=conv_strides) + conv1_bias
#* 输出特征图 shape: [None, 28, 28, 32]
conv1_conv = tf.nn.relu(output1)
#* 池化层 shape: [None, 14, 14, 32]
conv1_pool = max_pool_2x2(conv1_conv, ksize=kernal_size, strides=pool_strides)

#! 内部变量定义
#* 核尺寸大小
kernal_size = 3
#* 原始图片的深度
depth = 32
#* 滤波器的数量 也就是通道的数量
filter_num = 64
#* 卷积层步长
conv_strides = 1
#* 池化层步长
pool_strides = 2

#! conv2 & pool
#* conv shape: [ksize, ksize, in_size, out_size]
conv2_weight = get_weight([kernal_size, kernal_size, depth, filter_num])
conv2_bias = get_bias([filter_num])
#* 输出结果
output2 = conv2d(X_batch, conv2_weight, strides=conv_strides) + conv2_bias
#* 输出特征图 shape: [None, 28, 28, 32]
conv2_conv = tf.nn.relu(output2)
#* 池化层 shape: [None, 14, 14, 32]
conv2_pool = max_pool_2x2(conv2_conv, ksize=kernal_size, strides=pool_strides)
#* 扁平化池化层的输出 2d -> 1d
conv2_pool_flatten = tf.reshape(conv2_conv, [-1, conv2_pool.shape[0] * conv2_pool.shape[1] * conv2_pool.shape[2]])

#! fully connected1
#* 内部变量定义

fc1_weight = get_weight([7*7*64, 128])
fc1_bias = get_bias([128])
fc1_output = tf.nn.relu(tf.matmul(conv2_pool_flatten, fc1_weight) + fc1_bias)

#! fully connected1
#* 内部变量定义

fc2_weight = get_weight([128, 10])
fc2_bias = get_bias([10])
#* 分类问题 激活函数用 softmax
prediction_op = tf.nn.softmax(tf.matmul(fc1_output, fc2_bias) + fc2_bias)

loss_op = tf.losses.softmax_cross_entropy(onehot_labels=y_batch, logits=prediction_op)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)

init_op = tf.global_variables_initializer()

#! execute graph

with tf.Session() as session:
  session.run(init_op)
  for i in np.arange(step_num):
    xbatch, ybatch = mnist.train.next_batch(batch_size)
    xbatch = xbatch.reshape([batch_size, 28, 28, 1])
    _, loss = session.run([train_op, loss_op], feed_dict={X_batch: xbatch, y_batch: ybatch})
    print('loss is', loss)