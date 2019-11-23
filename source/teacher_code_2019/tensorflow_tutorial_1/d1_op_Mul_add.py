import tensorflow as tf

'''
# construct the 
a = tf.constant(2)
b = tf.constant(3)

#excute the graph
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))
    tf.summary.FileWriter("D:\\hello1",graph=tf.get_default_graph())
'''


#############################\
a = tf.constant(2)
b = tf.constant(3)

mat=tf.multiply(a,b)
add=tf.add(a,b)

#excute the graph
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(mat))
    print("Multiplication with constants: %i" % sess.run(add))
    tf.summary.FileWriter("D:\\hello1",graph=tf.get_default_graph())