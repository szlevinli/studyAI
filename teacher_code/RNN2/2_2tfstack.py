import tensorflow as tf 

a = tf.constant([1,2,3])
b = tf.constant([4,5,6])

c = tf.stack([a,b],axis=0)
print("the type of c is :",type(c))
print(c.shape)
d = tf.unstack(c,axis=0)
print("the type of d is :",type(d))
e = tf.unstack(c,axis=1)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
    



