from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MYMODEL(tf.keras.Model):
    def __init__(self):
        super(MYMODEL, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10)
    def call(self, inputs):
        output = self.dense1(inputs)
        return output

model_keras = MYMODEL()

input_spec = tf.TensorSpec([1, 64], tf.int32)
model_keras._set_inputs(input_spec, training=False)

# keras_input = tf.keras.Input([64], batch_size=1, dtype=tf.int32)
# keras_output = model_keras(keras_input, training=False)
# model_keras = tf.keras.Model(keras_input, keras_output)

print(model_keras.inputs)
print(model_keras.outputs)