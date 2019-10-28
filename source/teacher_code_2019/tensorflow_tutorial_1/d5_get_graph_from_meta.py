'''
#https://blog.csdn.net/wjc1182511338/article/details/82111790
#上一个方式适用于断点续训，且自己有构建图的完整代码，如果我要用别人的网络（fine tune），
#或者在自己原有网络上修改（即修改原有网络的某个部分），那么将网络的图重新构建一遍会很麻烦，
#那么我们可以直接从.meta文件中加载网络结构。
'''
import tensorflow as tf 
with tf.Session() as sess:
    #ckpt_state = tf.train.get_checkpoint_state('tmp/')  
    #restore  
    saver = tf.train.import_meta_graph('tmp/model.meta')
    graph = tf.get_default_graph()
    w3 = graph.get_tensor_by_name('W3:0')
    add1 = graph.get_tensor_by_name('add1:0')
    print(w3)
    #restore the varialbe
    saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
    print(sess.run(w3))
    #print(sess.run(tf.get_collection('W1')[0]))
    print("The process is finished")