from __future__ import print_function
import keras
import keras.datasets
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D
import random
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split


with open('chroma.pickle', 'rb') as f:
	data = pickle.load(f)
print(data.head(8))

x_train, x_test, y_train, y_test = train_test_split(list(data['bsa']), list(data['type']), test_size=0.08, random_state=42)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = keras.utils.to_categorical(y_train, 8)
y_test = keras.utils.to_categorical(y_test, 8)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


def create_model():
	v = Input(shape=(12,489,1,))
	vin = v
	v = Conv2D(12,(12,12))(v)
	#v = Conv2D(7,(1,21))(v)
	v = MaxPooling2D(pool_size = (1,36), strides=(1))(v)
	
	v = Flatten()(v)
	
	v = Dense(192, activation='elu')(v)
	v = BatchNormalization()(v)
	v = Dropout(0.2)(v)
	
	v = Dense(256, activation='elu')(v)
	v = BatchNormalization()(v)
	v = Dropout(0.2)(v)
	
	v = Dense(192, activation='elu')(v)
	v = BatchNormalization()(v)
	v = Dropout(0.2)(v)
	
	v = Dense(256, activation='elu')(v)
	v = BatchNormalization()(v)
	v = Dropout(0.2)(v)
	

	v = Dense(8, activation='softmax')(v)

	model = Model(inputs=vin, outputs=v)
	
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	print(model.summary())
	
	return model

model = create_model()

model.fit(x_train.flatten().reshape((x_train.shape[0], 12,489,1)), y_train,
          batch_size=256,
          epochs=30,
          verbose=1,
          validation_data=(x_test.flatten().reshape((x_test.shape[0], 12,489 ,1)), y_test))

#~ sys.exit(0)

x_train, x_test, y_train, y_test = train_test_split(list(data['bsa']), list(data['type']), test_size=0.01, random_state=1)

x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = keras.utils.to_categorical(y_train, 8)
y_test = keras.utils.to_categorical(y_test, 8)

score = model.evaluate(x_train.flatten().reshape((x_train.shape[0], 12,489 ,1)), y_train, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 


model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

sys.exit(0)
