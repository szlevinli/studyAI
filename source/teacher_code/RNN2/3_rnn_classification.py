import tensorflow as tf 
import numpy as np 
import input_data

#train parameters
step_num=1000
batch_size=300
lr = 0.1

#model parameters
n_steps= 28
n_inputs = 28
n_hidden = 128
class_num = 10

######################cunstruct graph###################

#X_batch = tf.placeholder(tf.float32, shape=[None,n_inputs])
X_batch = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
Y_batch = tf.placeholder(tf.float32, shape=[None,10])

weight = {
"in_W": tf.Variable(tf.truncated_normal(shape=[n_inputs,n_hidden])),
"out_W": tf.Variable(tf.truncated_normal(shape=[n_hidden,class_num]))
}

bias = {
"in_bias": tf.Variable(tf.constant(0.1,shape=[n_hidden])),
"out_bias": tf.Variable(tf.constant(0.1,shape=[class_num]))
}

def RNN(input,weight, bias):
    #input layer
    X = tf.reshape(input,shape=[-1, n_inputs])
    #h =  tf.nn.relu(tf.add(tf.matmul(X,weight["in_W"]),bias["in_bias"]))
    h = tf.add(tf.matmul(X,weight["in_W"]),bias["in_bias"])
    h1 = tf.reshape(h, shape=[-1,n_steps,n_hidden])
    
    #cell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    initial_state = rnn_cell.zero_state(batch_size,dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(rnn_cell, h1, initial_state= initial_state)

    #output layer
    #output = tf.nn.relu(tf.add(tf.matmul(state,weight["out_W"]),bias["out_bias"])
    output = tf.nn.softmax(tf.add(tf.matmul(state,weight["out_W"]),bias["out_bias"]))
    return output

prediction_op = RNN(X_batch, weight, bias) 
#loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=Y_batch, logits=prediction_op)
loss_op = tf.losses.softmax_cross_entropy(Y_batch, prediction_op)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss_op)
init_op = tf.global_variables_initializer()

mnist = input_data.read_data_sets("mnist_data",one_hot=True) 


######################execute graph###################
with tf.Session() as sess:
    sess.run(init_op)
    tf.summary.FileWriter("log",graph=tf.get_default_graph())
    for i in np.arange(step_num):
        xdata, ydata = mnist.train.next_batch(batch_size)
        xdata1=xdata.reshape([batch_size,n_steps, n_inputs])
        #print(xdata1.shape)
        _, loss = sess.run([train_op,loss_op],feed_dict={X_batch:xdata1,Y_batch:ydata})
        '''
        #print(loss)
        if i%100 == 0:
            pred = sess.run(prediction_op,feed_dict={X_batch:xdata1,Y_batch:ydata})
            result = tf.arg_max(pred, 1)
            label = tf.arg_max(ydata,1)
            #print(sess.run(result))
            #print(sess.run(label))
            accuracy = sess.run(tf.reduce_sum(tf.cast(tf.equal(result,label),tf.int32))/batch_size)
            print(accuracy)
        '''

        if i%100 == 0:
            test_data = mnist.test.images[:batch_size]
            test_reshape=test_data.reshape([batch_size,n_steps, n_inputs])
            pred = sess.run(prediction_op,feed_dict={X_batch:test_reshape, Y_batch:mnist.test.labels[:batch_size]})
            result = tf.arg_max(pred, 1)
            label = tf.arg_max(mnist.test.labels[:batch_size],1)
            #print(sess.run(result))
            #print(sess.run(label))
            accuracy = sess.run(tf.reduce_sum(tf.cast(tf.equal(result,label),tf.int32))/batch_size)
            print(accuracy)
