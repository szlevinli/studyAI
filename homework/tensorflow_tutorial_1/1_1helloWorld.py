
import tensorflow as tf

#python
#list, dictionary, 

#numpy
#numpy array



#constuct graph
hello = tf.constant('Hello, TensorFlow!')

#excute graph
#sess = tf.Session()
with tf.Session() as sess: 
    print(sess.run(hello))