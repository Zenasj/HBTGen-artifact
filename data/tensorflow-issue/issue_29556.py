from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

tf.enable_eager_execution()

sgd = tf.keras.optimizers.SGD()

inputs = tf.keras.Input(shape=(3,))

# First layer
x = tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

# Second layer
x = tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

# Output
outputs = tf.keras.layers.Dense(
	1,
	kernel_regularizer=tf.keras.regularizers.l2(0.01),
	activation='sigmoid'
)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=sgd, loss='mean_squared_error')

model.run_eagerly = True

# Define some dummy dataset
x = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
y = [0, 1, 0]

model.fit(x, y)