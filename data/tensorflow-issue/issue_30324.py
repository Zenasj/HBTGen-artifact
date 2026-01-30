from tensorflow import keras
from tensorflow.keras import layers

import psutil
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

for _ in range(100):
    tf.keras.Sequential([tf.keras.layers.Dense(3000, input_dim=3000)])
    print(psutil.virtual_memory().used / 2 ** 30)