from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf
import numpy as np

inp = tf.keras.layers.Input(shape=(1,))
out0 = tf.cast(inp, tf.int32)
out1 = tf.cast(inp, tf.float64)

model = tf.keras.Model(inputs=inp, outputs=[out0, out1])

model.compile(loss={model.output_names[1]: tf.losses.mse})

model.evaluate(
    np.ones((1, 1), dtype=np.float32),
    {model.output_names[1]: np.ones((1, 1), dtype=np.float32)},
)