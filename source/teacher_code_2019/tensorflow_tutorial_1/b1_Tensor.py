import tensorflow as tf

a = tf.constant([[1,2]])
b = tf.constant([[2],
                 [3]])

#c = a + b
c= tf.matmul(a,b)
print(c)



