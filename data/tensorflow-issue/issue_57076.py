from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from datetime import datetime

class LookupModel(tf.keras.Model):
    def __init__(self, table_size, name="lookup"):
        super().__init__(name=name)

        self.kvstore = tf.keras.layers.StringLookup(
            vocabulary=[str(i) for i in range(table_size)],
            trainable=False
        )

    def call(self, input):
        return input

small = LookupModel(10)
large = LookupModel(5_000_000)

small_start = datetime.now()
small([1])
print("Small Lookup:", datetime.now() - small_start)

large_start = datetime.now()
large([1])
print("Large Lookup:", datetime.now() - large_start)