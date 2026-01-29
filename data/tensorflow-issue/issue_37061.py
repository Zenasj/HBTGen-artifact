# tf.random.uniform((3, x)) where x varies since inputs nested: 
# OrderedDict with 'a': (3,1), 'b': (3,4), tuple with (3,3), (3,4) -> concatenated output shape: (3, 12)

import tensorflow as tf
import collections

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This layer concatenates the flattened nested inputs on last axis
        self.concat_layer = tf.keras.layers.Lambda(lambda x: tf.concat(tf.nest.flatten(x), axis=-1))

    def call(self, inputs):
        # inputs is a nested structure:
        # (OrderedDict({'a': Tensor(3,1), 'b': Tensor(3,4)}), (Tensor(3,3), Tensor(3,4)))
        # We flatten and concat all tensor inputs along last axis
        return self.concat_layer(inputs)

def my_model_function():
    # Returns an instance of MyModel, no weights needed
    return MyModel()

def GetInput():
    batch_size = 3

    # Construct the nested input matching the model input structure:
    # Tuple of (OrderedDict with keys 'a' and 'b'), and a tuple with two tensors
    input_values = (
        collections.OrderedDict((
            ('a', tf.random.uniform((batch_size, 1), dtype=tf.float32)),
            ('b', tf.random.uniform((batch_size, 4), dtype=tf.float32)),
        )),
        (
            tf.random.uniform((batch_size, 3), dtype=tf.float32),
            tf.random.uniform((batch_size, 4), dtype=tf.float32),
        ),
    )
    return input_values

