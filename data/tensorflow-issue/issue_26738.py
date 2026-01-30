from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

inputs = np.arange(10)
outputs = 2 * inputs

inner_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
inner_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError()])

outer_inputs = tf.keras.layers.Input(shape=[1])
outer_outputs = inner_model(outer_inputs)
outer_model = tf.keras.Model(inputs=outer_inputs, outputs=outer_outputs)
outer_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError()])
outer_model.evaluate(inputs, outputs) # crashes
outer_model.fit(inputs, outputs) # also crashes

outer_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanSquaredError(name="outer_model")])