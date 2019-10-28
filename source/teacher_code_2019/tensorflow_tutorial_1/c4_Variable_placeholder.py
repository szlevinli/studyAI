import tensorflow as tf

x_batch = tf.placeholder(tf.float32,shape=(None,1))
y_batch = tf.placeholder(tf.float32,shape=(None,1))

c = tf.add(x_batch, y_batch)

with tf.Session() as sess:
    result=sess.run(c,feed_dict={x_batch:[[1],[2],[3]],y_batch:[[2],[3],[4]]})
    print(result)