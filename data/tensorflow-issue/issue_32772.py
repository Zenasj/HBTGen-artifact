from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense = tf.keras.layers.Dense(4)

  def call(self, inputs, training=False):
    print('Training', training)
    return self.dense(inputs)
    
    
class Gen(tf.keras.utils.Sequence):
    def __len__(self):
        return 10
    
    def __getitem__(self, i):
        return np.ones((32, 100)), np.ones((32, 4))
    
model = MyModel()
model.compile(optimizer=tf.train.AdagradOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# training should be False
model.build(input_shape=(32, 100))

# training should be True
model.fit_generator(generator=Gen(), epochs=1, validation_data=Gen())