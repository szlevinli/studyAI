import tensorflow as tf

#sess = tf.InteractiveSession()

#construct the graph
inputTensor = tf.placeholder(tf.float32, [None, 784], name='inputTensor')
labelTensor=tf.placeholder(tf.float32, [None, 1], name='LabelTensor')
W = tf.Variable(tf.random_uniform([784, 1], -1.0, 1.0), name='weights')
b = tf.Variable(tf.zeros([1]), name='biases')
a = tf.nn.sigmoid(tf.matmul(inputTensor, W) + b, name='activation')

#tmp1=tf.multiply(inputTensor, W) 
#tmp2=tf.add(tmp1,b)

#execute the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(a)
    summary_writer = tf.summary.FileWriter('D:\\log1', tf.get_default_graph())

