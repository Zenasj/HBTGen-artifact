from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

self.state_size = NoDependency([TensorShape([self.units]),
                                        TensorShape([self.units, self.units])])

import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.keras.activations import softmax, relu
from tensorflow.python.keras.layers.recurrent import RNN, SimpleRNN, LSTM, GRU
from tensorflow.python.keras import activations, initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from tensorflow.keras.datasets import mnist

# set random seeds
RANDOM_SEED = 1
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = '0'

class BasicConfig(object):
    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 128
        self.units = 50

        self.output_dim = 10

        self.ac_alpha = 60
        self.ac_lambda = 0.105

        self.learning_rate = 0.01
        self.optimizer = Adam(self.learning_rate)
        self.loss_function = CategoricalCrossentropy()
        self.metric = CategoricalAccuracy()



class Model(Layer):
    def __init__(self, c, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.rnn_layer = RNN(AutoconceptorCell(c.units, alpha=c.ac_alpha,
                            lamb=c.ac_lambda))
        self.dense = Dense(c.output_dim, activation=softmax)

    def call(self, x):
        # TODO GITHUB line below causes error
        # x = rnn_layer(x, initial_state = [tf.zeros(self.units), tf.eye(self.units)])
        x = self.rnn_layer(x)
        output = self.dense(x)
        return output


class AutoconceptorCell(Layer):
    """
    Cell, very very similar to SimpleRNNCell, but added a matrix to the
    state, that is used in the call
    """

    def __init__(self,
               units,
               alpha,
               lamb,
               activation='tanh',
               kernel_initializer='glorot_uniform',
               recurrent_initializer='identity',
               bias_initializer='glorot_uniform',
               **kwargs):
        super(AutoconceptorCell, self).__init__(**kwargs)

        self.units = units

        self.aperture_fact = alpha**(-2)
        self.l = lamb 

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # IMPORTANT here i declare the state_size
        self.state_size = NoDependency([self.units,
                                            TensorShape([self.units, self.units])])

        self.output_size = self.units

        self.layer_norm = LayerNormalization(epsilon=1e-6)


    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)

        self.bias = self.add_weight(
            shape=(self.units,),
            name='bias',
            initializer=self.bias_initializer)

        self.built = True


    def call(self, inputs, states):
        # hidden state, and Conceptor matrix
        prev_h, C = states

        # input influence
        input_infl = K.bias_add(K.dot(inputs, self.kernel), self.bias)
        # recurrent influence
        recurrent_infl = K.dot(prev_h, self.recurrent_kernel)

        # h i.e. typical hidden layer update
        h = self.activation(input_infl + recurrent_infl)

        # update conceptor
        C = C + self.l * (K.batch_dot(tf.expand_dims(h - K.batch_dot(h, C, axes=(1,1)), 2), tf.expand_dims(h, 1)) - self.aperture_fact * C)

        # apply filter, then layer norm, then AGAIN activation
        h = self.activation(self.layer_norm(K.batch_dot(h, C, axes=(1,1))))
      
        return h, [h, C]



def get_mnist_data(batch_size=128, input_dim=28):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

    # transform target digits into one-hot-vectors
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # normalize input TODO why?
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # for single pixel mnist (i.e. minst minst_784) reshape input
    if (input_dim==1):
        x_train = x_train.reshape(x_train.shape[0], 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 28, 28)

     # bring into tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # bring into proper batch size
    train_dataset = train_dataset.batch(config.batch_size)
    test_dataset = test_dataset.batch(x_test.shape[0]) # test if full batch

    return (train_dataset, test_dataset)



if __name__ == '__main__':
    # config
    config = BasicConfig()
    # model with config
    model = Model(config)

    # data - minst
    train_dataset, test_dataset = get_mnist_data(batch_size=config.batch_size)

    for epoch_indx in range(config.num_epochs):
        # train
        for batch_indx, (x,t) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                output = model(x)

                gradients = tape.gradient(config.loss_function(t, output), model.trainable_variables)
            
            config.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # test
        for (x_test, t_test) in test_dataset:

            output_test = model(x_test)

        print("[E:{:3d}] Test Accuracy: {:04.5f}!".format(epoch_indx,
                                                    config.metric(t_test, output_test)))