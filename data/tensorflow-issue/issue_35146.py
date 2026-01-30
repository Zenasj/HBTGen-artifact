from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os

import tensorflow as tf

format_ext = ''  # '.h5' or empty for tf format
model_path = os.path.join('out', 'mnist-classifier{}'.format(format_ext))

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs = tf.keras.Input(shape=(784,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
                  # Loss function to minimize
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  # List of metrics to monitor
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.save(model_path)

import os

import tensorflow as tf

format_ext = ''  # '.h5' or empty for tf format
model_path = os.path.join('out', 'mnist-classifier{}'.format(format_ext))

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
)

(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    loaded_model = tf.keras.models.load_model(model_path)
    predictions = loaded_model.predict(x_test, batch_size=64)