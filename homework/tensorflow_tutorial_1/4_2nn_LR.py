import tensorflow as tf 
import numpy as np 

X1=np.linspace(-1,1,100)
X=np.reshape(X1,(100,1))
noisy=np.random.randn(100,1)
print(noisy)
Y=np.reshape(0.5*X+0.5+noisy, X.shape)

print(Y.shape)

X_batch = tf.placeholder(tf.float32, shape=(None,1))
Y_batch = tf.placeholder(tf.float32, shape=(None,1))

def nn_add_one_layer(input, insize, outsize,activation_function=None):
    W = tf.Variable(tf.truncated_normal((insize,outsize)),dtype=tf.float32)
    b = tf.Variable(tf.ones(outsize),dtype=tf.float32)
    if activation_function is None:
       return tf.nn.sigmoid(tf.matmul(input,W)+b)
    else:
       return activation_function(tf.matmul(input,W)+b)

h1=nn_add_one_layer(X_batch,1,10)
prediction=nn_add_one_layer(h1,10,1)

loss_op=tf.reduce_mean(tf.square(prediction-Y_batch))
train_op=tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in np.arange(50):
        #print(sess.run(prediction,feed_dict={X_batch:X, Y_batch:Y}))
        _,loss=sess.run([train_op,loss_op],feed_dict={X_batch:X, Y_batch:Y})
        print(loss)
        




