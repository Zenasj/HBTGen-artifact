# tf.random.uniform((1, 3, 64, 224, 224), dtype=tf.float32) ← Input shape and dtype inferred from the issue example

import tensorflow as tf
import functools
from collections.abc import Iterable

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters inferred from the example usage
        output_channels = 64
        kernel_shape = (7, 7, 7)
        is_training = True
        use_batch_norm = True
        activation_fn = 'relu'
        use_bias = False

        # Build the sub-layers pipeline similar to Unit3D layer
        self._pipeline_layers = []

        # Conv3D layer with channels_first format to match input shape (1,3,64,224,224)
        self._pipeline_layers.append(
            tf.keras.layers.Conv3D(
                filters=output_channels,
                kernel_size=kernel_shape,
                strides=(1, 1, 1),
                padding='same',
                use_bias=use_bias,
                data_format='channels_first'
            )
        )

        if use_batch_norm:
            # BatchNorm axis=1 to normalize channels_first format
            bn_layer = tf.keras.layers.BatchNormalization(axis=1, fused=False)
            # The issue's approach was to fix training flag using partial; here we fix it in call
            self._batch_norm = bn_layer
        else:
            self._batch_norm = None

        if activation_fn is not None:
            self._activation = tf.keras.layers.Activation(activation_fn)
        else:
            self._activation = None

        # Compose the pipeline functional layers (Conv -> BN -> Activation)
        # Using a list of callables, converted to a single composed callable in call method

    def call(self, inputs, training=None, mask=None):
        # Execute Conv3D
        x = self._pipeline_layers[0](inputs)
        # Execute BatchNorm if used, passing training flag properly
        if self._batch_norm is not None:
            x = self._batch_norm(x, training=training)
        # Execute activation
        if self._activation is not None:
            x = self._activation(x)
        return x

def my_model_function():
    # Return an instance of MyModel, no special initialization needed beyond __init__
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input of shape (1, 3, 64, 224, 224)
    # dtype float32 as per example from issue
    return tf.random.uniform(shape=(1, 3, 64, 224, 224), dtype=tf.float32)

