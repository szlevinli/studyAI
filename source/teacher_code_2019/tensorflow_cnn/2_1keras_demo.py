import tensorflow as tf
from keras.models import Sequantial
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

########Data process#
(train_data,train_label)(test_data,test_label)=mnist.load_mnist()
train_data = train_data.reshape(None,28,28,1)/225.0 
test_data = test_data.reshape(None,10)/225.0
train_label = to_categorical(train_label,num_class=10)
test_label = to_categorical(test_data,num_class=10)

#########
model = Sequantial()
model.add(Conv2D(32,))
model.add(Conv2D(batch_input_shape=(None, 1, 28, 28),filters=32,kernel_size=3,strides=1,padding='same',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first'))
model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding='same',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(10))
model.compile(train_op='Adam', loss='categorical_crossentropy')
model.fit(train_data,train_label,epoch=1,batch_size=10)