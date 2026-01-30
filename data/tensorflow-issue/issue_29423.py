from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow_datasets as tfds

# Disable eager execution, otherwise TPUStrategy won't work at all
tf.compat.v1.disable_eager_execution()

# Load dataset
ds = tfds.load('mnist', split=tfds.Split.TRAIN, as_supervised=True)
ds = ds.map(lambda x,y : (tf.cast(x, tf.float32), y))
ds = ds.shuffle(100).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Prepare strategy
import os
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tf.config.experimental_connect_to_host(TPU_ADDRESS)
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    net = tf.keras.Sequential([tf.keras.layers.Input([28,28,1]),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    net.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.SGD(0.01))

# training will raise an exception
net.fit(ds)