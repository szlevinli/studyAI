import tensorflow as tf

#construct the graph
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product=tf.matmul(matrix1,matrix2)

init_op = tf.global_variables_initializer()

#graph = tf.get_default_graph()
with tf.Session() as sess:
    #<op_name>:<output_index>
    sess.run(init_op)
    with tf.Graph.as_default() as graph:
        graph.get_tensor_by_name("product:0")