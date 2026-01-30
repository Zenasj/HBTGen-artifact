import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import smart_cond

import numpy as np

tf.config.optimizer.set_jit(True) # Enable XLA.

"""## Establish TPU access
(lifted from efficientnet example)
"""

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

"""## Keras class to demonstrate problem"""

class TPUBug(tf.keras.layers.Layer):

    def __init__(self, batchsize=32, **kwargs):
        super(TPUBug, self).__init__(**kwargs)
        self.batchsize = batchsize
  
    def build(self, input_shape):
        self.data_len = input_shape.as_list()[1]
        self.built = True

    def call(self, inputs):
        indices = [i for i in range(self.data_len)]
        indices = tf.convert_to_tensor(indices, dtype='int32')
        random_indices = tf.random.shuffle(indices)
        stacked = []
        for i in range(self.data_len):
            indexed = inputs[:, random_indices[i]]
            stacked.append(indexed)

        output = tf.stack(stacked, axis=-1)
        return output
  
    def compute_output_shape(self, input_shape):
        return input_shape

"""## Simple test"""

with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(17,))
    outputs = TPUBug(batchsize=32)(inputs)

    model = tf.keras.Model(inputs, outputs)
    # is this needed?
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

raw = np.arange(17, dtype='float32')
data = np.asarray([raw, raw + 100])
print('data:', data)

model.summary()

shuffled = model.predict(data)
print('shuffled:', shuffled)