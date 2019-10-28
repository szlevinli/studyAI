import tensorflow as tf
# Create some variables.
a = tf.Variable(tf.truncated_normal((2,3)),dtype=tf.float32,name="v1")
b = tf.Variable(tf.truncated_normal((3,2)),dtype=tf.float32,name="v2")
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# init=tf.global_variables_initializer()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # sess.run(init)
  # Restore variables from disk.
  saver.restore(sess, "model/model.ckpt")
  print("Model restored.")
  print(sess.run(b))
  # Do some work with the model