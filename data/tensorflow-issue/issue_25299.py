import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

tf.enable_eager_execution()

class MyModel(tf.keras.Model):
  def call(self, inputs):
    return inputs
  
model = MyModel()
model.compile(tf.keras.optimizers.Adam(), 'mean_squared_error')
inputs = labels = {'a': tf.range(5.0)}
model.fit(x=inputs, y=labels)

import tensorflow as tf

tf.enable_eager_execution()

class MyModel(tf.keras.Model):
  def call(self, inputs):
    return [inputs['a'], 2 * inputs['b']]
  
model = MyModel()
model.compile(tf.train.AdamOptimizer(), 'mean_squared_error')
inputs = {'a': tf.range(5.0), 'b': tf.range(5.0, 10.0)}
labels = [tf.range(5, 10), tf.range(5, 10)]
model.fit(x=inputs, y=labels)

loss = {"MyOutput1": "MyLoss"}
targets = {"MyOutput1": y1, "MyOutput2": y2}
outputs = {"MyOutput1": out1, "MyOutput2": out2}

loss = {"output_1": "MyLoss"}
targets = y1
outputs = [out1, out2]

import tempfile

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Dense(64)

    @tf.function(input_signature=[tf.TensorSpec((None, 128, 128), tf.float32)])
    def call(self, inputs):
        return self.layer(inputs)

model = MyModel()

batch_size = 16
inputs = np.random.random((batch_size, 128, 128)).astype('float32')
outputs = model(inputs)

print(inputs.shape, outputs.shape)

model_path = tempfile.mkdtemp()
tf.saved_model.save(model, model_path)

inputs = dict(input1=np.random.random(28, 28), input2=(16, 16, 3))
model.fit(inputs)

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Dense(64)

    @tf.function(input_signature=dict(input1=tf.TensorSpec((None, 128, 128), tf.float32),
                                      input2=tf.TensorSpec((None, 28, 28), tf.float32)))
    def call(self, inputs):
        return self.layer(inputs)

model = MyModel()

batch_size = 16
inputs = np.random.random((batch_size, 128, 128)).astype('float32')
outputs = model(inputs)