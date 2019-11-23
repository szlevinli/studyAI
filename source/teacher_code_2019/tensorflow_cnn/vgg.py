import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras.datasets import cifar10

#############Data proprecessing############


############Model#############
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=3, strides=1,padding='SAME'),batch_input_shape=(None,32,32,3))
model.add(MaxPooling2D(pool_size=(3,3), strides=1, padding='SAME'))
model.add(Conv2D(filters=32,kernel_size=3, strides=1,padding='SAME'),batch_input_shape=(None,32,32,3))
model.add(MaxPooling2D(pool_size=(3,3), strides=1, padding='SAME'))
flat = Flatten()
model.add()
