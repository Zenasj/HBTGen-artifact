from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_probability as tfp

tf.config.experimental_run_functions_eagerly(True)


def get_mnist_data(normalize=True, categorize=True):
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

    if categorize:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test, input_shape


def get_model(input_shape, num_classes=10):
    model = tf.keras.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(6, input_shape=input_shape, kernel_size=3, padding="SAME",
                                              activation=tf.nn.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tfp.layers.DenseFlipout(num_classes))
    return model


def train():
    x_train, y_train, x_test, y_test, input_shape = get_mnist_data()

    batch_size = 64

    model = get_model(input_shape)

    model.summary()

    model.compile(loss="categorical_crossentropy")

    model.fit(x_train, y_train, batch_size=batch_size, epochs=1)


if __name__ == '__main__':
    train()

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)