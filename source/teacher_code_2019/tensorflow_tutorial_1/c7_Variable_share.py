import tensorflow as tf 

with tf.variable_scope("variable") as scope:
    initializer = tf.constant_initializer(value=3)
    var1= tf.get_variable(name="var1",shape=[1], dtype=tf.float32, initializer=initializer)
    var2= tf.Variable([1],name="var2")
    scope.reuse_variables()
    var3= tf.Variable([2],name="var2")
    var4= tf.get_variable(name="var1")

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(var1.name)
    print(var2.name)
    print(sess.run(var2))
    print(var3.name)
    print(sess.run(var3))
    print(var4.name)
    print(sess.run(var4))



