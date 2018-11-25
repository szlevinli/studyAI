import tensorflow as tf
'''
sess = tf.InteractiveSession()
a = tf.Variable(0, name="a")
b = tf.Variable(1, name="b")
c = tf.add(a,b)

#sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('D:\\log', tf.get_default_graph())
summary_writer.close
'''

a = tf.Variable(0,name="var_a",dtype=tf.float16)
b = tf.Variable(1,name="var_1",dtype=tf.float16)

c = tf.add(a,b)
d = tf.multiply(a,b)
e = tf.div(c,d)

with tf.Session() as sess:
    rlt=sess.run(e, feed_dict={a:3, b:4})
    tf.summary.FileWriter("D:\\hello3",graph=tf.get_default_graph())
    print(rlt)
    


