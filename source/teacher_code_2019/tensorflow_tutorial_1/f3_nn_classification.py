import tensorflow as tf
import input_data
import numpy as np

step_num=2000
batch_size=1000

X_batch=tf.placeholder(tf.float32, shape=[None,784])
Y_batch=tf.placeholder(tf.float32, shape=[None,10])

data_sets = input_data.read_data_sets("mnist_data",one_hot=True)

def nn_add_one_layer(input, in_size, out_size,activation_function=None):
    #W = tf.Variable(tf.truncated_normal((insize,outsize)),dtype=tf.float32)
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    #b = tf.Variable(tf.ones(outsize),dtype=tf.float32)
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    if activation_function is None:
       #return tf.nn.sigmoid(tf.matmul(input,W)+b)
       return tf.matmul(input, W) + b
    else:
       return activation_function(tf.matmul(input,W)+b)

h1=nn_add_one_layer(X_batch,784,100,activation_function=tf.nn.sigmoid)

#method 1
prediction=nn_add_one_layer(h1,100,10,activation_function=tf.nn.softmax)
loss_op = tf.reduce_mean(-tf.reduce_sum(Y_batch* tf.log(prediction),reduction_indices=[1]))

#method 2
#prediction=nn_add_one_layer(h1,100,10)
#loss_op=tf.losses.softmax_cross_entropy(Y_batch,prediction)

#method 3
#loss_op = tf.losses.sparse_softmax_cross_entropy(labels=Y_batch, logits=prediction)

train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in np.arange(step_num):
        xdata,ydata =data_sets.train.next_batch(batch_size)
        _, loss =sess.run([train_op,loss_op],feed_dict={X_batch:xdata, Y_batch:ydata})
        
        if i%10==0:
            print(loss)

        if i%100 == 0:
            pred = sess.run(prediction,feed_dict={X_batch:data_sets.test.images, Y_batch:data_sets.test.labels})
            result = tf.argmax(pred, 1)
            label = tf.argmax(data_sets.test.labels,1)
            print(sess.run(result))
            print(sess.run(label))
            accuracy = sess.run(tf.reduce_mean(tf.cast(tf.equal(result,label),tf.float32)))
            print("the evaluation accuracy is ",accuracy)


    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))