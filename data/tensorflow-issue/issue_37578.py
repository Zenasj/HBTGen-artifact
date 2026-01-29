# tf.random.uniform((None, None, None, None), dtype=tf.float32)  # Input shape unknown from issue; using placeholder shape

import tensorflow as tf
import copy
from tensorflow.keras.layers import Wrapper
from tensorflow.keras.utils import serialize_keras_object, deserialize_keras_object

class MyModel(tf.keras.Model):
    """
    This model demonstrates the fixed behavior of the tf.keras.layers.Wrapper
    class serialization get_config and from_config methods as described in
    the issue.

    It encapsulates a wrapped layer and ensures the:
    1) Proper serialization of the wrapped layer using serialize_keras_object
    2) Usage of deep copy in from_config to avoid mutation of input config dict
    """

    def __init__(self, wrapped_layer=None, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # If no wrapped_layer is provided, default to a Dense layer for demonstration
        if wrapped_layer is None:
            wrapped_layer = tf.keras.layers.Dense(10)
        self.wrapper = Wrapper(wrapped_layer)

    def call(self, inputs, training=None):
        return self.wrapper(inputs)

    def get_config(self):
        # Corrected get_config to use serialize_keras_object on self.layer
        config = super(Wrapper, self.wrapper).get_config()
        # Build the config dict with properly serialized wrapped layer
        layer_config = serialize_keras_object(self.wrapper.layer)
        config.update({
            'layer': layer_config
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Use deepcopy to avoid side effects
        config_copy = copy.deepcopy(config)

        layer_config = config_copy.pop('layer')
        layer = deserialize_keras_object(layer_config)
        # Instantiate the wrapper with the deserialized layer and any other config keys
        return cls(wrapped_layer=layer, **config_copy)

def my_model_function():
    """
    Return an instance of MyModel. By default it wraps a Dense(10) layer.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor that can be input to MyModel.
    Given the example default wrapped layer is Dense(10), expecting shape (batch_size, features).
    Using batch size 4, features 20.
    """
    return tf.random.uniform((4, 20), dtype=tf.float32)

