from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

print('tf.version:', tf.__version__)  # 1.12.0

# tf.enable_eager_execution()  # try with eager or not

inputs = keras.Input(shape=(100,))
x = keras.layers.Dense(100,
                       kernel_initializer='zeros',
                       bias_initializer='ones',
                       trainable=False)(inputs)
x = keras.layers.Dropout(0.5)(x)
model = keras.Model(inputs, x)
model.compile(tf.train.RMSPropOptimizer(0.1), 'mse')

print(model.train_on_batch(np.ones((1, 100)), np.ones((1, 100))))  # Positive loss
model.fit(np.ones((1, 100)), np.ones((1, 100)))  # Positive loss

print(model.test_on_batch(np.ones((1, 100)), np.ones((1, 100))))  # 0 loss
model.evaluate(np.ones((1, 100)), np.ones((1, 100)))  # 0 loss