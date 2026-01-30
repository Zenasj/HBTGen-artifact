from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda

class TwoOutputs(tf.keras.layers.Layer):
    def call(self, x):
        return x + 1, x - 1

inputs = Input([2], dtype=tf.int32)
outputs = TwoOutputs()(inputs)  # subclass version is broken
# outputs = Lambda(lambda x: (x + 1, x - 1))(inputs)  # functional version also broken
model = tf.keras.Model(inputs, outputs)

print(model(tf.constant([[0, 1], [2, 3]])))  # works fine

tf.saved_model.save(model, 'checkpoints/test')
model = tf.saved_model.load('checkpoints/test')
infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
print(infer.structured_outputs)  # wrong signatures
print(infer(tf.constant([[0, 1], [2, 3]])))  # wrong output