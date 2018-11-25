#Inception Demo
import tensorflow as tf
import input_data
import numpy as np 

num_class=10
batch_size=100

slim = tf.contrib.slim

X_batch = tf.placeholder(tf.float32,shape=[None,28,28,1])
Y_batch = tf.placeholder(tf.float32,shape=[None,10])

def conv_2d(input,in_size,out_size,ksize,stride):
    weight = tf.Variable(tf.truncated_normal(shape=[ksize,ksize,in_size,out_size]))
    bias = tf.Variable(tf.constant(0.1,shape=[out_size]))
    #the dimension of conv is [None, 28, 28, out_size]//线性变换
    conv=tf.nn.conv2d(input,weight,[1,stride,stride,1],padding="SAME")
    #the dimension of ouput is [None, 28, 28, out_size]//非线性变换
    output=tf.nn.relu(tf.add(conv,bias))
    return output

def max_pool(input,ksize,stride):
    return tf.nn.max_pool(input,[1,ksize,ksize,1],[1,stride,stride,1],padding="SAME")

def inception_block(input):    
    #branch_0 = tf.nn.conv2d(input,****)
    #branch_1 = tf.nn.conv2d(input,****)
    branch_0 = slim.conv2d(input,16,[1,1])
    branch_1 = tf.contrib.slim.conv2d(input,16,[3,3])
    branch_2 = tf.contrib.slim.conv2d(input,16,[5,5])
    #branch_3 = tf.contrib.slim.max_pool2d(input,16,[3, 3])
    return tf.concat(axis=3,values=[branch_0, branch_1, branch_2])

#conv2d
net = conv_2d(X_batch,1,32,3,1)   #stride = 1
net = max_pool(net,2,2) #stride=2

#Inception
net = inception_block(net)
print(net)
slim.max_pool2d(net, [3, 3],stride=2)

#global pooling,
#net = slim.avg_pool2d(net, [7, 7], stride=1)
net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
logits = slim.conv2d(net,num_class,[1,1])

loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=Y_batch, logits=logits)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

init_op = tf.global_variables_initializer()

mnist = input_data.read_data_sets("mnist_data",one_hot=True)

with tf.Session() as sess:
    sess.run(init_op)
    for i in np.arange(500):
        xs, ys = mnist.train.next_batch(batch_size)
        xs_reshape=xs.reshape([batch_size,28,28,1])
        _,loss = sess.run([train_op, loss_op], feed_dict={X_batch:xs_reshape,Y_batch:ys})

        if i%100 == 0:
            test_data = mnist.test.images[:1000]
            test_reshape=test_data.reshape([1000,28,28,1])
            #pred = sess.run(logits,feed_dict={X_batch:xs_reshape, Y_batch:ys})
            #print(pred.shape)  #(100, 1, 1, 10)
            #pred1 = sess.run(tf.squeeze(logits,1),feed_dict={X_batch:xs_reshape, Y_batch:ys})
            #print(pred1.shape)  #(100, 1, 10)
            pred2 = sess.run(tf.squeeze(logits,[1,2]),feed_dict={X_batch:test_reshape, Y_batch:mnist.test.labels[:1000]})
            #print(pred2.shape)  #(100, 10)
            result = tf.arg_max(pred2, 1)
            label = tf.arg_max(mnist.test.labels[:1000],1)
            accuracy = sess.run(tf.reduce_sum(tf.cast(tf.equal(result,label),tf.int32))/batch_size)
            print(accuracy)
    