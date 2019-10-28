import tensorflow as tf
import input_data
import numpy as np 

num_class=10
batch_size=100

slim = tf.contrib.slim

X_batch = tf.placeholder(tf.float32,shape=[None,28,28,1])
Y_batch = tf.placeholder(tf.float32,shape=[None,10])

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))

def biase_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv_2d(input,in_size,out_size,ksize,stride):
    weight = tf.Variable(tf.truncated_normal(shape=[ksize,ksize,in_size,out_size]))
    bias = tf.Variable(tf.constant(0.1,shape=[out_size]))
    #the dimension of conv is [None, 28, 28, out_size]//线性变换
    conv=tf.nn.conv2d(input,weight,[1,stride,stride,1],padding="SAME")
    #the dimension of ouput is [None, 28, 28, out_size]//非线性变换
    output=tf.nn.relu(tf.add(conv,bias))
    return output

def max_pool_2x2(input,ksize,stride):
    return tf.nn.max_pool(input,[1,ksize,ksize,1],[1,stride,stride,1],padding="SAME")


#conv1
#W_conv1=weight_viriable([3,3,1,32])   #28*28*1
#b_conv1=bias_viriable([32])
#h_conv1=tf.nn.relu(conv2d(X_batch,W_conv1)+b_conv1)
h_conv1 = conv_2d(X_batch,in_size=1,out_size=32,ksize=3,stride=1)
h_pool1=max_pool_2x2(h_conv1,ksize=3,stride=2)    #14*14*1

#conv1
#W_conv2=weight_viriable([3,3,32,64])  
#b_conv2=bias_viriable([64])
#h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_conv2 = conv_2d(h_pool1,in_size=32,out_size=64,ksize=3,stride=1)
h_pool2=max_pool_2x2(h_conv2,ksize=2,stride=2)    #7*7*1
 
#fc1
W_fc1=weight_variable([7*7*64,1024])
b_fc1=biase_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,shape=[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#fc2
W_fc2=weight_variable([1024,10])
b_fc2=biase_variable([10])
prediction_op = tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)

loss_op = tf.losses.softmax_cross_entropy(onehot_labels=Y_batch,logits=prediction_op)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

init_op = tf.global_variables_initializer()

mnist = input_data.read_data_sets("mnist_data",one_hot=True)

with tf.Session() as sess:
    sess.run(init_op)
    for i in np.arange(1000):
        xs, ys = mnist.train.next_batch(batch_size)
        xs_reshape=xs.reshape([batch_size,28,28,1])
        _,loss = sess.run([train_op, loss_op], feed_dict={X_batch:xs_reshape,Y_batch:ys})

        if i%100 == 0:
            test_data = mnist.test.images[:1000]
            test_reshape=test_data.reshape([1000,28,28,1])
            pred = sess.run(prediction_op,feed_dict={X_batch:test_reshape, Y_batch:mnist.test.labels[:1000]})
            result = tf.arg_max(pred, 1)
            label = tf.arg_max(mnist.test.labels[:1000],1)
            #print(sess.run(result))
            #print(sess.run(label))
            accuracy = sess.run(tf.reduce_sum(tf.cast(tf.equal(result,label),tf.int32))/batch_size)
            print(accuracy)
    







