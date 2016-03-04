import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet

dataset = pd.read_csv("Data/train.csv")
y_train = dataset[[0]].values.ravel()
x_train = dataset.iloc[:, 1:].values
test = pd.read_csv("Data/test.csv")

y_train = y_train.astype(np.uint8)
x_train = np.array(x_train).reshape((-1,1,28,28)).astype(np.uint8)
test = np.array(test).reshape((-1,1,28,28)).astype(np.uint8)

def c_n_net(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 28, 28),

    conv1_num_filters=7,
    conv1_filter_size=(3, 3),
    conv1_nonlinearity=lasagne.nonlinearities.rectify,

    pool1_pool_size=(2, 2),

    conv2_num_filters=12,
    conv2_filter_size=(2, 2),
    conv2_nonlinearity=lasagne.nonlinearities.rectify,

    hidden3_num_units=1000,
    output_num_units=10,
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1


cnn = c_n_net(20).fit(x_train, y_train)

pred = cnn.predict(test)

np.savetxt('mnist_nn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')