import tensorflow as tf
#from keras.activations import relu
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

#Data processing
(train_data, train_label),(test_data, test_label)=mnist.load_data()
train_data = train_data.reshape(train_data.shape[0],-1)/255
test_data = test_data.reshape(test_data.shape[0],-1)/255
train_label = to_categorical(train_label,num_classes=10)
test_label = to_categorical(test_label,num_classes=10)

#Build Model
model = Sequential()
dense1 = Dense(10, input_shape=(784,),activation='relu')
model.add(dense1)
dense2 = Dense(10,activation='softmax')
model.add(dense2)

######model train################
model.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_label, batch_size=64, epochs=3)
