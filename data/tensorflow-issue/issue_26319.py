from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense = tf.keras.layers.Dense(4)

  def call(self, inputs, training=False):
    print('Training', training)
    return self.dense(inputs)
    
model = MyModel()
model.compile(optimizer=tf.train.AdagradOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

inp = np.ones((5, 3), dtype=np.float32)
out = np.ones((5, 4), dtype=np.float32)

# training should be False
model(inp)

# training should be True
model.fit(inp, out)