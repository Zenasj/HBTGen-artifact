import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ConstantTensorLayer(Layer):
   def __init__(self, data, **kwargs):
      super(ConstantTensorLayer, self).__init__(**kwargs)
      self.data = tf.constant(data)

   def call(self, x):
      return tf.keras.layers.Reshape((-1,), name='Reshaped_ConstTensorLayer')(self.data)

   def get_config(self):
        config = super().get_config().copy()
        config.update({
            'data': self.data,
        })
        return config

class ConstantTensorLayer(Layer):
   def __init__(self, data, **kwargs):
      super(ConstantTensorLayer, self).__init__(**kwargs)

      # Save the initializer's input as-is
      self.data = data

      # Create/store native tensors as desired
      self.tensor = tf.constant(data)

   def call(self, x):
      return tf.keras.layers.Reshape((-1,), name='Reshaped_ConstTensorLayer')(self.data)

   def get_config(self):
        config = super().get_config().copy()

        config.update({
             # Return the original input --+
             #                             |
             #            +----------------+
             #            |
             #            v
            'data': self.data,
        })

        return config