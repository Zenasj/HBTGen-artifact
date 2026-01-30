import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


def train(file_path):
    print('Loading data ...')
    x_train = np.float32(np.random.randint(-8, 8, size=[1254521, 56, 40]))
    y_train = [[1, 0] for _ in range(x_train.shape[0])]
    y_train = np.float32(y_train)

    print(x_train.shape, y_train.shape)

    print('Create model ...')
    data = tf.keras.Input(shape=[56, 40])
    x = tf.keras.layers.Reshape([56, 40, 1])(data)

    cnn_out = tf.keras.layers.Conv2D(8, (20, 5), strides=(1, 2))(x)
    cnn_output = tf.keras.layers.Reshape([37, 18 * 8])(cnn_out)

    gru_out = tf.keras.layers.GRU(20)(cnn_output)

    outputs = tf.keras.layers.Dense(2)(gru_out)

    my_model = tf.keras.Model(inputs=data, outputs=outputs)

    my_model.compile()

    print(my_model.summary())

    my_model.fit(x_train, y_train, epochs=100, batch_size=1024, validation_split=0.1)
    my_model.save(file_path)


if __name__ == '__main__':
    train("test.h5")