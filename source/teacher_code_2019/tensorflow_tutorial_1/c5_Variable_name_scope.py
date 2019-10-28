import tensorflow as tf 

with tf.name_scope("name") as scope:
    var1= tf.get_variable("var1",initializer=[1])
    var2= tf.Variable([1],name="var2")
    var3= tf.Variable([2],name="var2")
    var4= tf.Variable([3],name="var2")

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



