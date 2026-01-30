import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

loaded = tf.keras.models.load_model('my_model')
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

# output
{'output_1': TensorSpec(shape=(None, 96, 96, 1), dtype=tf.float32, name='output_1')}

class MyModel(tf.keras.models.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.conv = tf.keras.layers.Conv2D(
        1,
        3,
        padding='same',
    )

  @tf.function(input_signature=[tf.TensorSpec([None, None, None, 1])])
  def call(self, inputs):
    return self.conv(inputs)