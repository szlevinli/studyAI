import tensorflow as tf 

a = tf.constant([[1,2]])
b = tf.constant([[2],
                 [3]])
c= tf.matmul(a,b)

'''
# method 1
sess = tf.Session()
print("the result is :", sess.run(c))
sess.close()
'''

#method 2
with tf.Session() as sess:
    print("the result is :", sess.run(c))
    

