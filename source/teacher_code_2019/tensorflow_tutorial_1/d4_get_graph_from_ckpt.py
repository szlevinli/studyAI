import tensorflow as tf 

#Method 1: 必须先build graph，然后再restore，不然就会报错
#不然报出的错误是ValueError: No variables to save
def build_graph():
    w1 = tf.Variable([1,3,10,15],name='W1',dtype=tf.float32)
    w2 = tf.Variable([3,4,2,18],name='W2',dtype=tf.float32)
    w3 = tf.placeholder(shape=[4],dtype=tf.float32,name='W3')
    w4 = tf.Variable([100,100,100,100],dtype=tf.float32,name='W4')
    add = tf.add(w1,w2,name='add')
    add1 = tf.add(add,w3,name='add1')
    return w3,add1

with tf.Session() as sess:
    ckpt_state = tf.train.get_checkpoint_state('tmp/')
    #w3,add1=build_graph()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)

