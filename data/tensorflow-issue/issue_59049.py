from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import layers

input_img= layers.Input(shape=(112, 112, 3))

x = [cnn_example()(input_img) for _ in range(n_repeats)]
x = layers.Lambda(lambda x: tf.stack(x, axis=1))(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(1)(x)

model = tf.keras.Model(input_img, x)