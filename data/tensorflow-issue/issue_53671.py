from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
tf.debugging.set_log_device_placement(True)
        
_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
x = _input
with tf.device("/GPU:1"):
    x = tf.keras.layers.Dense(10, name="should_be_on_gpu")(x)
    x = tf.keras.layers.Dense(10, name="should_be_on_gpu_2")(x)
model = tf.keras.models.Model(inputs=[_input], outputs=[x])
model.compile('adam', 'mse')
model.summary()
model.fit([2], [4])

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise ValueError("At least one GPU required for this test!")
if len(gpus) == 1:
    # Create two virtual GPUs for this test:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
          tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"{len(gpus)} physical GPUs, split into {len(logical_gpus)} logical GPUs")
    print(logical_gpus)

_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)

with tf.device("/GPU:0"):
    x = _input
    x = tf.keras.layers.Dense(10, name="should_be_on_first_gpu")(x)
    x = tf.keras.layers.Dense(10, name="should_also_be_on_first_gpu")(x)
    gpu0 = x
with tf.device("/GPU:1"):
    x = _input
    x = tf.keras.layers.Dense(10, name="should_be_on_second_gpu")(x)
    x = tf.keras.layers.Dense(10, name="should_also_be_on_second_gpu")(x)
    gpu1 = x

model = tf.keras.models.Model(inputs=[_input], outputs=[gpu0, gpu1])
model.compile('adam', 'mse')
model.summary()
model.fit([2], [4])