# tf.random.uniform((B, 128, 128, 1), dtype=tf.float32) ← Inferred input shape from original example

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def max_pool_with_argmax(x):
    """
    Max pool operation with argmax indices returned as described.
    This is required to pass indices for the custom unpooling layer.
    """
    return tf.nn.max_pool_with_argmax(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        include_batch_in_index=True  # Include batch in index for safe unpooling
    )

class UnpoolingLayer(Layer):
    """
    Custom unpooling layer using saved argmax indices from max-pooling.
    This layer is used in SegNet and DeconvNet models to perform
    max-unpooling by scattering the pooled values back to their original locations.
    """
    def __init__(self, pooling_argmax, stride=[1, 2, 2, 1], **kwargs):
        super(UnpoolingLayer, self).__init__(**kwargs)
        self.pooling_argmax = pooling_argmax
        self.stride = stride

    def build(self, input_shape):
        super(UnpoolingLayer, self).build(input_shape)

    def call(self, inputs):
        # Shape components as int64 tensors
        input_shape = K.cast(K.shape(inputs), dtype='int64')

        # Compute output shape after unpooling (height and width multiplied by stride)
        output_shape = (input_shape[0],
                        input_shape[1] * self.stride[1],
                        input_shape[2] * self.stride[2],
                        input_shape[3])

        argmax = self.pooling_argmax

        # Create a tensor of ones with same shape as argmax for broadcasting
        one_like_mask = K.ones_like(argmax, dtype='int64')

        # Range of batch indices for scatter_nd indexing shaped for broadcasting
        batch_range = K.reshape(
            K.arange(start=0, stop=input_shape[0], dtype='int64'),
            shape=[input_shape[0], 1, 1, 1]
        )

        # Broadcast batch indices to full tensor shape
        b = one_like_mask * batch_range

        # Calculate spatial indices from flattened argmax indices
        y = argmax // (output_shape[2] * output_shape[3])
        x = (argmax % (output_shape[2] * output_shape[3])) // output_shape[3]

        # Channel indices for scatter_nd
        feature_range = K.arange(start=0, stop=output_shape[3], dtype='int64')
        f = one_like_mask * feature_range

        # Number of updates = total elements in inputs
        updates_size = tf.size(inputs)

        # Stack indices (b, y, x, f) for scatter_nd, then transpose to shape (N, 4)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))

        # Flatten inputs to 1D for scatter updates
        values = K.reshape(inputs, [updates_size])

        # Use scatter_nd to place inputs values in unpooled tensor shape
        return tf.scatter_nd(indices, values, output_shape)

    def compute_output_shape(self, input_shape):
        # Output shape doubles spatial dims according to stride
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])

    def get_config(self):
        base_config = super(UnpoolingLayer, self).get_config()
        base_config.update({
            'pooling_argmax': self.pooling_argmax,
            'stride': self.stride
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MyModel(tf.keras.Model):
    """
    MyModel implements a minimal test of the custom UnpoolingLayer 
    using max_pool_with_argmax as in SegNet/DeconvNet.
    It accepts a 4D input tensor (B, 128, 128, 1), applies max-pooling with indices,
    a Conv2D layer, then custom unpooling by the saved indices, matching the example.
    """

    def __init__(self):
        super(MyModel, self).__init__()

        # First conv layer after pooling
        self.conv1 = Conv2D(64, kernel_size=3, padding='same', 
                            kernel_initializer='he_normal', name='stage1_conv1')

        # We will store argmax indices after pooling internally
        self.pooling_argmax = None

    def call(self, inputs):
        # Perform max pool with argmax (returns tuple: pooled, indices)
        pool, argmax = max_pool_with_argmax(inputs)
        self.pooling_argmax = argmax

        # Conv layer on pooled output
        x = self.conv1(pool)

        # Unpool using the stored argmax indices
        unpool = UnpoolingLayer(self.pooling_argmax, name='unpool1')(x)

        # Optional second conv after unpooling (commented out in original example)
        # x = self.conv1(unpool)  # reuse conv1 weights, or define new conv layer if desired

        # Return unpooled output tensor
        return unpool

def my_model_function():
    """
    Instantiate and return the MyModel model.
    """
    return MyModel()

def GetInput():
    """
    Generate a valid random input tensor for MyModel.
    Shape: (Batch, Height, Width, Channels) = (1, 128, 128, 1)
    Values: uniform random floats
    """
    # Batch size 1 chosen for simplicity/introspection; can be any int >= 1
    return tf.random.uniform((1, 128, 128, 1), dtype=tf.float32)

