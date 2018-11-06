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
batch_size = 512
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
model = Sequential()

print(filepath)
modelExist = os.path.exists(filepath)
if(modelExist == True):
	model = load_model(filepath)
else:

	model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(SpatialDropout2D(0.25))

	# model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
model.summary()

	# Display evaluate
score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

	# Learning 
checkpointer=ModelCheckpoint(filepath=filepath, verbose=1, monitor="val_acc", save_best_only=True)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

model.fit(x_train, y_train,
			epochs=epochs,
			batch_size=batch_size,
			verbose=1,
			shuffle=True,
			validation_data=(x_test, y_test),
			callbacks=[tensorboard, checkpointer])
score=model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

