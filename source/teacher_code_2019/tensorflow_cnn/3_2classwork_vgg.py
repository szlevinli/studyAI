import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.optimizers import SGD

#import data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#用于正则化时权重降低的速度
weight_decay = 0.0005
nb_epoch=100
batch_size=32

#layer1 32*32*3
model = Sequential()
##############################
model.add(Conv2D(***, (***, ***), padding='same',
input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))

##############################
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##############################
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##############################
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##############################
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(***, (***, ***), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##############################
model.add(Flatten())
model.add(Dense(***))
model.add(Activation('relu'))
model.add(Dense(***))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(***))
model.add(Activation('softmax'))

##############################
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=nb_epoch, batch_size=batch_size,
             validation_split=0.1, verbose=1)

model.save('my_model_bp.h5')