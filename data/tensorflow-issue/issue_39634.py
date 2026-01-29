# tf.random.uniform((B, H, W, C), dtype=tf.float32) <- assuming generic 4D input typical for dropout

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops, math_ops, array_ops
from tensorflow.python.keras.utils import tf_utils

class MyModel(tf.keras.Model):
    def __init__(self, rate=0.5, seed=None):
        super(MyModel, self).__init__()
        # This model encapsulates a customized DropoutControl layer
        self.dropout = DropoutControl(rate=rate, seed=seed)

    def call(self, inputs, training=False):
        # Forward pass uses DropoutControl layer
        return self.dropout(inputs, training=training)

class DropoutControl(Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super(DropoutControl, self).__init__(**kwargs)
        self.rate = rate
        self.seed = seed
        self.cache_dropout_mask = None

    def reset_dropout(self):
        # Called to reset (cache) new dropout mask
        rate = ops.convert_to_tensor(
              self.rate, dtype=self.input_dtype, name="rate")
        # Use shape stored previously (should match input shape)
        random_tensor = random_ops.random_uniform(
            shape=self.shape, seed=self.seed, dtype=self.input_dtype)
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        keep_mask = random_tensor >= rate
        self.cache_dropout_mask  = scale * math_ops.cast(keep_mask, self.input_dtype)

    def get_dropout_mask(self):
        # Return cached mask for external querying if needed
        return self.cache_dropout_mask 

    def call(self, inputs, training=False):
        # On first call, cache input shape and dtype, then reset dropout mask
        if self.cache_dropout_mask is None:
            self.shape = array_ops.shape(inputs)
            self.input_dtype = inputs.dtype
            self.reset_dropout()

        def dropped_inputs():
            return inputs * self.cache_dropout_mask

        # Use smart_cond to apply dropout mask only in training mode
        output = tf_utils.smart_cond(training,
                                     dropped_inputs,
                                     lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'seed': self.seed
        }
        base_config = super(DropoutControl, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def my_model_function():
    # Return an instance of MyModel with default dropout rate and no seed
    return MyModel()

def GetInput():
    # Return a random input tensor to match expected input shape for MyModel.
    # The original dropout example assumes a generic 4D shape (batch, height, width, channels).
    # Use standard float32 input with shape (2, 32, 32, 3) - e.g. like an image batch.
    return tf.random.uniform((2, 32, 32, 3), dtype=tf.float32)

