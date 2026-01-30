import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

SHAPE = (1, 5)

class TestModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense_layer = tf.keras.layers.Dense(10)

    @tf.function(input_signature=[tf.TensorSpec(shape=SHAPE, dtype=tf.float32)])
    def run(self, x):
        return self.dense_layer(x)


module = TestModel()
sample_input = tf.random.normal(SHAPE, dtype=tf.float32)
module.run(sample_input)

np.save('sample_input.npy', sample_input.numpy())
tf.saved_model.save(module, "test_model")

# # To reproduce, run the following:
# python test.py && saved_model_cli run --dir test_model --tag_set serve --signature_def serving_default --inputs 'x=sample_input.npy'