from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyDense(tf.keras.layers.Layer):
  def __init__(self, num_units, **kwargs):
    super(MyDense, self).__init__(**kwargs)
    self.num_units = num_units

  def build(self, input_shape):
    kernel_shape = [input_shape[-1], self.num_units * 2, self.num_units]
    bias_shape = [self.num_units]

    self.kernel = self.add_weight("kernel", shape=kernel_shape, trainable=True)
    self.bias = self.add_weight("bias", shape=bias_shape, trainable=True)
    super(MyDense, self).build(input_shape)

  def call(self, inputs):
    return tf.einsum("ac,cde->ade", inputs, self.kernel) + self.bias

inputs = tf.keras.Input(shape=(10,), dtype=tf.float32)
outputs = MyDense(15)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("SUCCESS!")

converter.experimental_new_converter = True
converter.target_spec.supported_ops =[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]