import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#X1=np.linspace(-1,1,100)
#X=np.reshape(X1,[100,1])
#noisy=np.random.randn(100,1)
#Y=np.reshape(10*X+10+noisy, X.shape)

X = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, X.shape).astype(np.float32)
Y = np.square(X) - 0.5 + noise

#plt.show()

#X_batch = tf.placeholder(tf.float32, shape=(None,1))
#Y_batch = tf.placeholder(tf.float32, shape=(None,1))

X_batch = tf.placeholder(tf.float32, [None, 1])
Y_batch = tf.placeholder(tf.float32, [None, 1])

def nn_add_one_layer(input, insize, outsize,activation_function=None):
    #W = tf.Variable(tf.truncated_normal((insize,outsize)),dtype=tf.float32)
    W = tf.Variable(tf.random_normal([insize, outsize]))
    #b = tf.Variable(tf.ones(outsize),dtype=tf.float32)
    b = tf.Variable(tf.zeros([1, outsize]) + 0.1)
    if activation_function is None:
       #return tf.nn.sigmoid(tf.matmul(input,W)+b)
       return tf.matmul(input, W) + b
    else:
       return activation_function(tf.matmul(input,W)+b)


# add hidden layer
#h1=nn_add_one_layer(X_batch,1,10)
h1 = nn_add_one_layer(X_batch, 1, 10, activation_function=tf.nn.relu)
# add output layer
#prediction=nn_add_one_layer(h1,10,1)
prediction = nn_add_one_layer(h1, 10, 1, activation_function=None)


#loss_op=tf.reduce_mean(tf.square(prediction-Y_batch))
#loss_op=tf.nn.l2_loss(prediction-Y_batch)
loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - Y_batch),
                     reduction_indices=[1]))
train_op=tf.train.GradientDescentOptimizer(0.1).minimize(loss_op)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in np.arange(100):
        _,loss =sess.run([train_op,loss_op],feed_dict={X_batch:X, Y_batch:Y})
        print(loss)
        if i == 99:
            result=sess.run(prediction,feed_dict={X_batch:X})

    print(result.shape)
    plt.scatter(X,Y)
    plt.plot(X,result)
    plt.show()



