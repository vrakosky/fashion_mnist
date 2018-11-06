import os
import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Lambda, SpatialDropout2D

epochs = 25
num_classes = 10
batch_size = 300
input_shape = (28, 28, 1)
filepath = "cnn_model_best_94.2.hdf5"

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

#------------- Checkpoint
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, monitor="val_acc", save_best_only=True)

#------------- Tensorboard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#------------- Reduce Learning Rate Function
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', min_lr=0.00000001)

#------------- Data Augmentation
gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.15, shear_range=.1, fill_mode='nearest')

#------------- MODEL
model = Sequential()
modelExist = os.path.exists(filepath)
if(modelExist == True):
	model = load_model(filepath)
else:
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(SpatialDropout2D(0.25))

	model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
	model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(SpatialDropout2D(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
model.summary()

#------------- Display evaluate
score=model.evaluate(x_test, y_test, verbose=0)

#------------- Display first iteration result
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for i in range(0, 5):
	#------------- Fit
	model.fit(x_train, y_train,
				epochs=epochs,
				batch_size=batch_size,
				verbose=1,
				shuffle=True,
				validation_data=(x_test, y_test),
				callbacks=[tensorboard, checkpointer, reduce_lr])

	#------------- Reinitinilize learning rate & Reload best model
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
	model = load_model(filepath)

	#------------- Fit Generator
	batches = gen.flow(x_train, y_train, batch_size=batch_size)
	model.fit_generator(batches, 
						steps_per_epoch=len(x_train)/batch_size, 
						epochs=epochs, 
						validation_data=(x_test,y_test),
						verbose=1,
						shuffle=True,
						validation_steps=10000//batch_size,
						callbacks=[tensorboard, checkpointer, reduce_lr])

	#------------- Display last iteration result
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


#------------- Tensorboard
#- python -m tensorboard.main --logdir="\logs"


# - ASTUCE
# - Augmenter le batch
# - Réduire le learning rate si il n'y a aucun changement : ReduceLROnPlateau
