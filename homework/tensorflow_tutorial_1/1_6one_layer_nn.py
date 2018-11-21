import tensorflow as tf

inputTensor = tf.placeholder(tf.float32, [None, 784], name='inputTensor')
labelTensor=tf.placeholder(tf.float32, [None, 1], name='LabelTensor')
W = tf.Variable(tf.random_uniform([784, 1], -1.0, 1.0), name='weights')
b = tf.Variable(tf.zeros([1]), name='biases')


predict_op = tf.nn.sigmoid(tf.matmul(inputTensor, W) + b, name='activation')

loss_op = tf.nn.l2_loss(predict_op - labelTensor, name='L2Loss')

train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_op)

#sess.run(tf.global_variables_initializer())
#summary_writer = tf.summary.FileWriter("log3", sess.graph)

#sess.run(tf.global_variables_initializer())
#summary_writer = tf.summary.FileWriter('D:\\log2', tf.get_default_graph())
#summary_writer.close

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #_,loss=sess.run([loss_op, train_op],feed_dict=)
    summary_writer = tf.summary.FileWriter('D:\\log2', sess.graph)
