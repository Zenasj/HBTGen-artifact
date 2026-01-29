# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assuming 4D tensor with batch and spatial dims

import tensorflow as tf
from tensorflow import keras

class ParametricScalar(keras.layers.Layer):
    """
    ParametricScalar layer: Learns a set of scalar weights (alpha)
    that are multiplied element-wise to the input tensor.
    Similar in spirit to PReLU's learnable parameter, but this layer
    simply scales inputs by learned parameters.
    
    Arguments:
      alpha_initializer: initializer for the alpha weights (default 'ones')
      shared_axes: axes along which to share the same scalar parameter
                   i.e. the alpha parameter shape is 1 along these axes.
                   For example, if input shape is (B, H, W, C),
                   shared_axes=(1,2) will share scalar for each channel across spatial dims.
    
    Input shape:
      Arbitrary shape, but typically 4D tensor (B, H, W, C)
    
    Output shape:
      Same as input shape.
    """
    def __init__(self, alpha_initializer='ones', shared_axes=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_initializer = keras.initializers.get(alpha_initializer)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        # input_shape is a tf.TensorShape, input_shape[0] is batch size
        param_shape = list(input_shape[1:])  # exclude batch dim
        if self.shared_axes is not None:
            for i in self.shared_axes:
                # shared_axes are axes indices relative to input, exclude batch dim
                param_shape[i - 1] = 1
        self.alpha = self.add_weight(
            shape=param_shape,
            name='alpha',
            initializer=self.alpha_initializer,
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.alpha

    def get_config(self):
        config = {
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'shared_axes': self.shared_axes
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# To follow the instructions, wrap this ParametricScalar layer in a tf.keras.Model class named MyModel

class MyModel(tf.keras.Model):
    """
    Model that applies a ParametricScalar layer on the input.
    """
    def __init__(self, shared_axes=None):
        super().__init__()
        # Use default initializer 'ones' so initial behavior is identity scaling
        self.param_scalar = ParametricScalar(shared_axes=shared_axes)

    def call(self, inputs):
        return self.param_scalar(inputs)


def my_model_function():
    # Return an instance of MyModel with example shared_axes
    # For standard 4D input (B, H, W, C), sharing over spatial dims (1,2) makes sense
    return MyModel(shared_axes=[1, 2])


def GetInput():
    # Generate random float32 input matching a typical 4D input tensor shape:
    # Batch=4, Height=32, Width=32, Channels=3 (e.g., image batch)
    return tf.random.uniform((4, 32, 32, 3), dtype=tf.float32)

