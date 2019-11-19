import tensroflow as tf 
import numpy as np 
import input_data as mnist

###########construct graph###
X_input = tf.placeholder(tf.float32,shape=[None,28,28,1])
Y_input = tf.placeholder(tf.float32,shape=[None,10])

def add_conv_layer(X,input_feature_num,filter_num,kernel_size,stride,activaction_function=None):
    W = tf.Variable(tf.truncated_normal(shape=[input_feature_num,filter_num, kernel_size,kernel_size]))
    b = tf.Variable(tf.constants(0.1,shape=[filter_num]))
    Wx_plus_b = tf.nn.conv2(X,W,[1,stride,stride,1],padding='SAME')
    if activaction_function is None:
        return Wx_plus_b
    else：
        activaction_function(Wx_plus_b)

def add_pool_layer(X,pool_size,stride):
    return tf.nn.max_pool(X,[1,pool_size,pool_size,1],[1,stride,stride,1],padding="SAME")

def add_nn_layer(X,in_size,out_size,activaction_function=None):
    W = tf.Variable(tf.truncated_normal(shape=[in_size,out_size]))
    b=tf.Variable(tf.constants(0.1,shape=[out_size]))
    Wx_plus_b = tf.matmul(X,W) + b
    if activaction_function is None:
        return Wx_plus_b
    else:
        return activaction_function(Wx_plus_b)
#conv1:[None,28,28,32]
conv1=add_conv_layer(X_input,1,32,3,1,tf.nn.relu)
#pool1:[None,14,14,32]
pool1=add_pool_layer(conv1,3,2)
#conv2:[None,14,14,64]
conv2=add_conv_layer(pool1,32,64,3,1,tf.nn.relu)
#pool2:[None,7,7,64]
pool2=add_pool_layer(con2,3,2)
flatten_layer=tf.reshape(pool2,shape=[-1,7*7*64])
nn1 = add_nn_layer(flatten_layer,7*7*64,1024,tf.nn.relu)
output = add_nn_layer(nn1,1024,10)

loss_op = tf.losses.softmax_cross_entropy(Y_input,output)
train_op = tf.train.GradientDescentOptimizer(lr=0.01).minimize(loss)

init_op = tf.global_variables_initializer()

##########executre graph######
with tf.Session() as sess:
    sess.run(init_op)
    for i in :