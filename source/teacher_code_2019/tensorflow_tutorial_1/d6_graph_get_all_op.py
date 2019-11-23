import tensorflow as tf

#construct the graph
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product=tf.matmul(matrix1,matrix2)

graph = tf.get_default_graph()
with tf.Session() as sess:
    for op in graph.get_operations():
        print(op.name)