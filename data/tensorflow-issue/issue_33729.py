from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

# tf.compat.v1.disable_eager_execution()

def get_bayesian_model(input_shape=None, num_classes=10):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tfp.layers.Convolution2DFlipout(6, kernel_size=5, padding="SAME", activation=tf.nn.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tfp.layers.DenseFlipout(84, activation=tf.nn.relu))
    model.add(tfp.layers.DenseFlipout(num_classes))
    return model

def get_mnist_data(normalize=True):
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if normalize:
        x_train /= 255
        x_test /= 255

    return x_train, y_train, x_test, y_test, input_shape


def train():
    # Hyper-parameters.
    batch_size = 128
    num_classes = 10
    epochs = 1

    # Get the training data.
    x_train, y_train, x_test, y_test, input_shape = get_mnist_data()

    # Get the model.
    model = get_bayesian_model(input_shape=input_shape, num_classes=num_classes)

    # Prepare the model for training.
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # Train the model.
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.evaluate(x_test, y_test, verbose=0)


if __name__ == "__main__":
    train()