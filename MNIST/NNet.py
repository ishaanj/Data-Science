import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

dataset = pd.read_csv("Data/train.csv")
y_train = dataset[[0]].values.ravel()
x_train = dataset.iloc[:, 1:].values
test = pd.read_csv("Data/test.csv")

y_train = y_train.astype(np.uint8)
x_train = np.array(x_train).reshape((-1,1,28,28)).astype(np.uint8)
test = np.array(test).reshape((-1,1,28,28)).astype(np.uint8)

n_net = NeuralNet(
                layers = [('input', layers.InputLayer),('hidden', layers.DenseLayer),('output', layers.DenseLayer)],
                input_shape = (None, 1, 28, 28),
                hidden_num_units = 1000,
                output_nonlinearity = lasagne.nonlinearities.softmax,
                output_num_units = 10,
                update = nesterov_momentum,
                update_learning_rate = 0.0001,
                update_momentum = 0.9,
                max_epochs = 15,
                verbose = 1
        )

n_net.fit(x_train, y_train)

pred = n_net.predict(test)

np.savetxt('mnist_nn.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')