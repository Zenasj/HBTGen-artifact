# tf.random.uniform((B, None, 3), dtype=tf.float32)  # Input shape with unknown time steps, 3 features

import tensorflow as tf

class SetShapeLayer(tf.keras.layers.Layer):
    """
    Custom layer to apply set_shape to its input tensor during the call.
    This encapsulates the set_shape operation so it can be serialized and
    saved in the model. Since set_shape is a no-op that modifies the 
    static shape only, wrap it in a Layer for saving/loading support.
    """
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        # shape is a tuple or list representing the static shape to enforce,
        # e.g. (None, 2, 3)
        self._fixed_shape = shape

    def call(self, inputs):
        # Apply set_shape to inputs.
        inputs.set_shape(self._fixed_shape)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self._fixed_shape})
        return config


class MyModel(tf.keras.Model):
    """
    Fused model demonstrating the original issue and the common workaround.
    Accepts input of shape (batch_size, None, 3).
    Applies set_shape via a custom layer to enforce known static shape.
    This fixes the post-load shape issue that arises when using `set_shape` 
    directly on a tensor outside a Layer.
    Then applies a Dense layer.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom layer that sets shape to (None, 2, 3),
        # matching the example where output shape had unknown "None" replaced by 2.
        self.set_shape_layer = SetShapeLayer((None, 2, 3))
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        # inputs shape: (batch_size, None, 3)
        x = self.set_shape_layer(inputs)  # enforce known shape inside layer
        return self.dense(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random float32 tensor matching expected input shape:
    # batch size 4, unknown sequence length 5 (picked arbitrarily), 3 features
    # This shape matches the example Input layer in the issue.
    return tf.random.uniform(shape=(4, 5, 3), dtype=tf.float32)

