import tensorflow as tf 

a = tf.Variable(tf.truncated_normal((2,3)),dtype=tf.float32,name="a")
b = tf.Variable(tf.truncated_normal((3,2)),dtype=tf.float32,name="b")

c = tf.matmul(a,b)

#Thie sentence is very easy to be forgettend
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)    
    print("the result is :", sess.run(c))