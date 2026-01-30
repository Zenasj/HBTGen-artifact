from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)
print(tf.config.list_physical_devices(), flush=True)


tf.compat.v1.disable_eager_execution()

inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(1)(inp)

model = tf.keras.Model(inp, out)

try:
    model.predict(
        np.zeros((32, 1)),
        callbacks=[tf.compat.v1.keras.callbacks.TensorBoard(log_dir="test")],
    )
except Exception as e:
    print("Failed! Error:", str(e), flush=True)
else:
    print("Success!", flush=True)