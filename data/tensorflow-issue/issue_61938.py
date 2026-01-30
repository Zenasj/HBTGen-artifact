import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
"""
XLA compiled
"""
class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(64, 128)

    @tf.function(jit_compile=True)
    def call(self, x1, x2):
        x3 = self.embedding(x1)
        return (x3 * x2)

tf.random.set_seed(42)
m = Model()
input_1 = tf.constant([64], dtype=tf.int32)
input_2 = tf.constant([[[[10.0]]]], dtype=tf.float32)
print(m(input_1, input_2))

"""
Without XLA
"""
class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(64, 128)

    def call(self, x1, x2):
        x3 = self.embedding(x1)
        return (x3 * x2)

tf.random.set_seed(42)
m = Model()
input_1 = tf.constant([64], dtype=tf.int32)
input_2 = tf.constant([[[[10.0]]]], dtype=tf.float32)
print(m(input_1, input_2))

"""
InvalidArgumentError: Exception encountered when calling layer 'embedding_4' (type Embedding).

{{function_node __wrapped__ResourceGather_device_/job:localhost/replica:0/task:0/device:CPU:0}} indices[0] = 64 is not in [0, 64) [Op:ResourceGather] name: 
"""