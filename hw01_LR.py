import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

n_samples = 1000 # 样本数量
n_features = 1  # 特征数量

#* 创建特征数据集
#? shape(n_samples, n_features) 元素为0到10之间的均分数字
x = np.linspace(-1, 1, n_samples * n_features).reshape(n_samples, n_features)
# x = 10 * np.random.random((n_samples, n_features))
# x = np.random.normal(0, 1, (n_samples, n_features))

#* 创建标签
# noise = np.random.randn(n_samples, 1)
noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
y = np.square(x) - 0.5 + noise
#y = np.sum(y, axis=1)
#y = y[:, np.newaxis]

#* 创建批次
x_batch = tf.placeholder(tf.float32, [None, 1])
y_batch = tf.placeholder(tf.float32, [None, 1])

#* 添加神经网络隐藏层
#@param input 输入数据
#@param insize 输入的 node 数
#@param outsize 输出的 node 数
#@param activation_function 激活函数
def add_nn_one_layer(input, insize, outsize, activation_function=None):
  W = tf.Variable(tf.random_normal([insize, outsize]))
  b = tf.Variable(tf.zeros([1, outsize]) + 0.1)

  if activation_function is None:
    return tf.matmul(input, W) + b
  else:
    return activation_function(tf.matmul(input, W) + b)

h1 = add_nn_one_layer(x_batch, 1, 10, activation_function=tf.nn.relu)
#h2 = add_nn_one_layer(h1, 5, 5, activation_function=tf.nn.relu)
prediction = add_nn_one_layer(h1, 10, 1, activation_function=None)




#* loss function is MAE
loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - y_batch),
                          reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

init_op = tf.global_variables_initializer()

step_num = 100

with tf.Session() as session:
  session.run(init_op)
  for i in np.arange(step_num):
    _, loss = session.run([train_op, loss_op], feed_dict={x_batch: x, y_batch: y})
    print('loss = ', loss)
    # if i == (step_num - 1):
    #   result = session.run(prediction, feed_dict={x_batch: x})

  result = session.run(prediction, feed_dict={x_batch: x})

  print(result.shape)
  plt.scatter(x, y)
  plt.plot(x, result, color='r')
  plt.show()
