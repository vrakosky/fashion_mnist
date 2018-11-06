import os
import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Lambda, SpatialDropout2D

#------------- Setting network parameters
epochs = 1
num_classes = 10
batch_size = 300
input_shape = (28, 28, 1)
filepath = "cnn_model_best.hdf5"

#------------- Dataset of 60,000 28x28 training images & 10,000 28x28 test images.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#------------- Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#------------- Model
model = Sequential([
    Dense(512, input_shape=(784,), activation='relu'),
    Dense(128, activation = 'relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer=Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=20,
                    verbose=1,
                    validation_data=(X_val, y_val))


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])