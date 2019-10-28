import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

#training paramters
batch_start= 0
batch_size = 20
step_num = 200

#model parameters
n_input = 1
n_step = 20
n_hidden = 10
n_output = 1

def get_batch():
    global batch_start, n_step
    # xs shape (50batch, 20steps)
    xs = np.arange(batch_start, batch_start+n_step*batch_size).reshape((batch_size, n_step)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    batch_start += n_step
    #plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    #plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# input:
# output:predict_op, loss_op, train_op
# share variable

class LSTM(object):
    def __init__(self, n_input,n_step,n_hidden,n_output,batch_size):
        self.n_input = n_input
        self.n_step = n_step
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.batch_size = batch_size
        with tf.name_scope("input"):
            self.X_batch = tf.placeholder(tf.float32, shape=[None, n_step, n_input], name="X_batch")
            self.Y_batch = tf.placeholder(tf.float32, shape=[None, n_step, n_output], name="Y_batch")
        with tf.variable_scope("input_layer"):
            self.add_input_layer()
        with tf.variable_scope("cell_layer"):
            self.add_cell()
        with tf.variable_scope("output_layer"):
            self.add_output_layer()
        with tf.name_scope("loss"):
            self.add_loss()
        with tf.name_scope("optimizer"):
            self.add_optimizer()
        
    def add_input_layer(self):
        X=tf.reshape(self.X_batch, shape=[-1,self.n_input])
        W_in = self._add_weight_(shape=[self.n_input,self.n_hidden])
        b_in = self._add_bias_(shape=[self.n_hidden])
        h=tf.add(tf.matmul(X, W_in), b_in)
        self.h=tf.reshape(h,shape=[-1, self.n_step, self.n_hidden])
 
    def add_cell(self):
        #cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size,dtype=tf.float32)
        self.output, self.state = tf.nn.dynamic_rnn(cell, self.h, initial_state=init_state,time_major=False)

    def add_output_layer(self):
        # output is a list, the dimension of  output is (batchsize, n_step, n_hidden)
        Y = tf.reshape(self.output, shape=[-1,n_hidden])
        W_out = self._add_weight_(shape=[n_hidden,n_output])
        b_out = self._add_bias_(shape=[n_output])
        #the dimension of state[1] is 2-dimension (batchsize, n_hidden)
        #self.prediction_op = tf.add(tf.matmul(self.state[1],W_out),b_out)
        self.prediction_op = tf.add(tf.matmul(Y,W_out),b_out)


    def add_loss(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.prediction_op, [-1], name='reshape_pred')],
            [tf.reshape(self.Y_batch, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_step], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )

        self.cost = tf.reduce_sum(losses)/tf.to_float(self.batch_size)   
    
    def add_optimizer(self):
        self.train_op=tf.train.AdamOptimizer(0.006).minimize(self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _add_weight_(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,) 
        return tf.get_variable(name=name,shape=shape,initializer=initializer)
        
    def _add_bias_(self, shape, name='bias'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name,shape=shape,initializer=initializer)   

model = LSTM (n_input, n_step, n_hidden, n_output, batch_size)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    plt.ion()
    plt.show()
    for i in np.arange(step_num):
        seq, res, xs =get_batch()
        _,loss = sess.run([model.train_op, model.cost],feed_dict={model.X_batch:seq,model.Y_batch:res})
        print(loss)
        pred = sess.run(model.prediction_op,feed_dict={model.X_batch:seq,model.Y_batch:res})
        #the dimension of predition is (1000,1)
        #print(pred.shape)
        #plt.plot(xs[0, :], pred[0, :], 'r', xs[0, :], seq[0, :], 'b--')
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:n_step], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)


            