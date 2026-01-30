import random
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

input_layer = keras.Input(shape=(3, 4))
X = layers.Dense(10)(input_layer)
X = layers.Dense(1)(X)[..., 0]      #      <-----------  with ellipsis
model = keras.Model(input_layer, X)
loss = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()
model.compile(optimizer, loss=loss)
model.fit(
    np.random.random((10, 3, 4)).astype(np.float32),
    np.ones((10, 3)).astype(np.float32),
    epochs=10,
    batch_size=5,
)
tflite_model_multi = tf.lite.TFLiteConverter.from_keras_model(
    model
)
tflite_model_multi = tflite_model_multi.convert()

with open('my_model.tflite', 'wb') as fin:
    fin.write(tflite_model_multi)
interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
print(interpreter.get_output_details())
interpreter.allocate_tensors()
print(interpreter.get_output_details())

...
X = layers.Dense(1)(X)[:, :, 0]      #      <-----------  without ellipsis
...