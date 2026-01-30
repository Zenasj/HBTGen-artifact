from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

def gaussian(x, amp=1, mu=None, sig=None):
    """ Gaussian function over d dimensions of x
    """
    if mu is None:
        mu = np.zeros_like(x)
    if sig is None:
        sig = np.ones_like(x)
    return amp * np.exp(-np.sum(np.square(x - mu) / (2 * np.square(sig))))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


multiplier = np.zeros((28,), dtype=np.float32)
for i in range(4):
    multiplier[13-i] = round(gaussian(i),2)
    multiplier[14+i] = round(gaussian(i),2)
print(multiplier)

x_train = np.einsum('bhw,d->bdhw', x_train, multiplier)[...,np.newaxis]


class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, ksizes, strides, shape):
        super(ExtractPatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides
        self.shape = shape

    def call(self, inputs):
        patches = tf.extract_volume_patches(inputs,
                                        ksizes=self.ksizes,
                                        strides=self.strides,
                                        padding="VALID")
        return tf.reshape(patches, self.shape), tf.shape(inputs)

class CombinePatches(tf.keras.layers.Layer):
    def __init__(self, ksizes, strides):
        super(CombinePatches, self).__init__()
        self.ksizes = ksizes
        self.strides = strides

    def call(self, patches, inputs):
        target_volume = tf.zeros_like(inputs)
        target_patches = tf.extract_volume_patches(
            target_volume,
            ksizes=self.ksizes,
            strides=self.strides,
            padding="VALID"
        )
        # Creates list of gradient mappings from patches to target shape
        # Patches without overlap get 1, elements that overlap receive 1 
        # times the number of overlaps.
        target_grad_mapping = tf.gradients(target_patches, target_volume)[0]

        # Computes gradients again and dividing by grad, otherwise its just summed.
        return tf.gradients(target_patches, target_volume, patches)[0] / target_grad_mapping


def create_model():
    inputs = tf.keras.layers.Input(shape=(None,None,None,1))
    patches, shape = ExtractPatches(ksizes=[1,14,14,14,1], strides=[1,14,14,14,1], shape=(-1,14,14,14,1))(inputs)
    encoded = tf.keras.layers.Conv3D(filters=28, kernel_size=(14,14,14), strides=(14,14,14))(patches)
    decoded = tf.keras.layers.Conv3DTranspose(filters=1, kernel_size=(14,14,14), strides=(14,14,14))(encoded)
    merged = CombinePatches(ksizes=[1,14,14,14,1], strides=[1,14,14,14,1])(decoded, inputs)

    return tf.keras.models.Model(inputs=inputs, outputs=merged)

ae = create_model()

ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
           loss=tf.keras.losses.MeanSquaredError(),
           metrics=['accuracy'])


test_history = ae.fit(x_train,
                       x_train,
                       batch_size=1,
                       epochs=1,
                       callbacks=None)