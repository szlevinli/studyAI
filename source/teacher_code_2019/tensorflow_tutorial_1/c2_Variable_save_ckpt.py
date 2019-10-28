'''
.meta文件保存了当前图结构
.index文件保存了当前参数名
.data文件保存了当前参数值
'''

import tensorflow as tf
# Create some variables.
#v1 = tf.Variable('v1', name="v1")
#v2 = tf.Variable('v2', name="v2")

a = tf.Variable(tf.truncated_normal((2,3)),dtype=tf.float32,name="v1")
b = tf.Variable(tf.truncated_normal((3,2)),dtype=tf.float32,name="v2")

#Thie sentence is very easy to be forgettend
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Save the variables to disk.
  save_path = saver.save(sess, "model/model.ckpt")
  print("Model saved in file: ", save_path)