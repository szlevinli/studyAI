import numpy as np 
import tensorflow as tf 
from keras.datasets import mnist
from keras.utils import to_categorical
from ops import *
import time


##################################################################################
# Data proprecess
##################################################################################
def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_data = train_data/255.0
    test_data = test_data/255.0

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    return train_data, train_labels, test_data, test_labels


class ResNet(object):
    def __init__(self, sess, checkpoint_dir='checkpoint',log_dir='logs',res_n=18, epoch=10, batch_size=32,lr=0.1):
        self.model_name = 'ResNet'
        self.sess = sess
        self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
        self.img_size = 28
        self.c_dim = 1
        self.label_dim = 10

        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.res_n = res_n

        self.epoch = epoch
        self.batch_size = batch_size
        self.iteration = len(self.train_x) // self.batch_size

        self.init_lr = lr


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(self.test_x), self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(self.test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_inptus)
        self.test_logits = self.network(self.test_inptus, is_training=False, reuse=True)

        self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss


        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)


    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        epoch_lr = self.init_lr
        start_epoch = 0
        start_batch_id = 0
        counter = 1

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x,
                    self.test_labels : self.test_y
                }


                # update network
                _, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
  
                # test
                test_loss, test_accuracy = self.sess.run(
                    [self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)


                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0


def main():
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cnn = ResNet(sess)
        # build graph
        cnn.build_model()
        cnn.train()


if __name__ == '__main__':
    main()