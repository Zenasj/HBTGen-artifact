import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
from scipy.linalg import dft
from math import sqrt

# Layer that does tf.signal.fft operation
class FFTLayer(tf.keras.layers.Layer):

    def call(self, x):
      fx = tf.signal.fft(x)
      return fx

# Layer that returns same results as tf.signal.fft op, but
# uses slower direct computation of DFT, implemented as matrix multiply.
class MatrixDFTLayer(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
      self.dft = tf.cast(dft(1024), tf.complex64)

    def call(self, x):
        fx = self.dft @ tf.transpose(x)
        return tf.transpose(fx)


def create_model(use_mirrored_strategy: bool = True,
                              run_eagerly: bool = True,
                              layer_to_use: tf.keras.layers.Layer = FFTLayer) -> None:
    print(f"\ncreate model with: use_mirrored_strategy: {use_mirrored_strategy}, ",
          f"run_eagerly: {run_eagerly}, ",
          f"layer_to_use: {layer_to_use}")

    if use_mirrored_strategy:
        distribution_strategy = tf.distribute.MirroredStrategy()
    else:
        distribution_strategy = tf.distribute.get_strategy()
    with distribution_strategy.scope():

        ins = tf.keras.layers.Input([1024], dtype=tf.complex64)
        x = layer_to_use()(ins)
        model = tf.keras.Model(inputs=ins, outputs=x)

        model.compile(
            loss=tf.keras.losses.MeanAbsoluteError(),
            run_eagerly=run_eagerly
        )
    return model


def create_data(fft_size, batch_size, num_steps):
    num_examples = num_steps * batch_size

    # y data is a complex vector of all (1/sqrt(2), (1/sqrt(2)j)
    train_y = np.ones([fft_size], np.float32)
    train_y = (1/sqrt(2))*train_y + (1/sqrt(2))*1j*train_y
    # abs mean is 1 -> MAE magnitude should be compared to 1
    print("train_y mean: ", tf.reduce_mean(tf.abs(train_y)))

    # use inverse transform to create input data
    # fft(train_x) will produce train_y
    train_x = tf.signal.ifft(train_y)

    # clone data to get larger training set
    train_y = train_y[tf.newaxis, ...]
    train_x = train_x[tf.newaxis, ...]

    train_x = tf.tile(train_x, [num_examples, 1])
    train_y = tf.tile(train_y, [num_examples, 1])
    
    return train_x, train_y


fft_size = 1024
batch_size = 9
num_steps = 100
train_x, train_y = create_data(fft_size, batch_size, num_steps)

# Test cases with MatrixDFTLayer
# These are all ok, MAE close to 0.0
# ok
model = create_model(use_mirrored_strategy=False, run_eagerly=False, layer_to_use=MatrixDFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")
# ok
model = create_model(use_mirrored_strategy=False, run_eagerly=True, layer_to_use=MatrixDFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")
# ok
model = create_model(use_mirrored_strategy=True, run_eagerly=False, layer_to_use=MatrixDFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")

# Test Cases using TF FFT. These fail when using MirroredStrategy.
# ok
model = create_model(use_mirrored_strategy=False, run_eagerly=False, layer_to_use=FFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")
# ok
model = create_model(use_mirrored_strategy=False, run_eagerly=True, layer_to_use=FFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")
# fail, 
model = create_model(use_mirrored_strategy=True, run_eagerly=False,layer_to_use= FFTLayer)
loss = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
print(f"loss: {loss}")

import numpy as np
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

print("multi gpu mirrored strategy test sanity check:")

def run_fft(img):
    print(tf.signal.fft2d(tf.experimental.numpy.copy(img)))

strategy = tf.distribute.MirroredStrategy()
rand_mat = tf.random.uniform(
                (1, 1, 686, 686),
                minval=0.0,
                maxval=1.0,
                dtype=tf.dtypes.float32,
                seed=None
            )
img = tf.cast(rand_mat, tf.complex64)
with strategy.scope():
    strategy.run(run_fft, args=(img,))