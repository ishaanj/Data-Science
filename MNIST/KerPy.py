import pandas as pd
import time as ts
import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D

#Parameters
NB_FILTERS = 32
NB_POOL = 2
NB_CONVOLUTION = 3
BATCH_SIZE = 16
NB_CLASSES = 10
NB_EPOCH = 12

#Read input
dataset = pd.read_csv("Data/train.csv")
y_train = dataset[[0]].values.ravel().astype(np.float32)
x_train = dataset.iloc[:, 1:].values
test = pd.read_csv("Data/test.csv")

#Reshape input
y_train = np_utils.to_categorical(y_train)
x_train = x_train.reshape(x_train.shape[0],1,28,28).astype(np.float32)
x_train /= 255
test = np.array(test).reshape(-1,1,28,28).astype(np.float32)
test /= 255

#Model Architecture
model = Sequential()
model.add(Convolution2D(NB_FILTERS, NB_CONVOLUTION, NB_CONVOLUTION,
                        border_mode='valid',
                        input_shape=(1, 28, 28)))
model.add(Activation('relu'))
model.add(Convolution2D(NB_FILTERS, NB_CONVOLUTION, NB_CONVOLUTION))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(NB_POOL, NB_POOL)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

#Start Timer
time_start = ts.time()
time_start = dt.datetime.fromtimestamp(time_start).strftime("%H:%M:%S")
print("Start Train: ", time_start)

#Model Main
model.compile(loss='categorical_crossentropy', optimizer='Adadelta')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, show_accuracy=True, verbose=1, validation_split=0.1)

#End Timer
time_end = ts.time()
time_end = dt.datetime.fromtimestamp(time_end).strftime("%H:%M:%S")
print("End Train: ", time_end)

#predict and write
print("Predicting...")
pred = model.predict_classes(test, batch_size=BATCH_SIZE, verbose=1)
np.savetxt('Results/mnist_keras3.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
