# tf.random.uniform((3, 224, 224), dtype=tf.float32) ← inferred input shape and type from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A TensorFlow model wrapper that accepts a custom tensor-like object.
    Since subclassing tf.Tensor is not supported, this model expects
    an input object that wraps a tf.Tensor and exposes a `.tensor` property.
    The model simply forwards the inner tensor to the underlying layers.

    This design mirrors the recommended pattern in the issue:
    do NOT subclass tf.Tensor directly (immutable and unsupported),
    but create a wrapper class which is tensor-like.
    """

    def __init__(self):
        super().__init__()
        # For demonstration, a simple conv layer to process inputs
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu"
        )
        # For demonstration, a global pooling after convolution
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None):
        # Extract the underlying tensor from the MyTensor wrapper
        # Assumes input is an instance of MyTensor (defined below)
        x = inputs.tensor  # unwrap to get true tf.Tensor
        x = self.conv(x)
        x = self.pool(x)
        return x

def my_model_function():
    # Return an instance of MyModel; no special weights initialization needed here
    return MyModel()

class MyTensor:
    """
    A custom tensor-like wrapper class that holds a tf.Tensor internally.
    This approach is recommended instead of subclassing tf.Tensor.
    This class exposes a .tensor attribute, and can add custom methods or properties.
    """

    def __init__(self, tensor: tf.Tensor):
        if not isinstance(tensor, tf.Tensor):
            raise TypeError("MyTensor expects a tf.Tensor instance")
        self._tensor = tensor

    @property
    def tensor(self):
        # Provide access to the internal tf.Tensor
        return self._tensor

    # Example of forwarding attribute access to inner tensor (optional)
    def __getattr__(self, name):
        # Delegate attribute access to underlying tf.Tensor if not found on MyTensor
        return getattr(self._tensor, name)

def GetInput():
    # Return a MyTensor wrapping a tf.Tensor of shape (3, 224, 224) with float32 dtype
    # This input is compatible with MyModel, which expects MyTensor inputs
    shape = (3, 224, 224)
    tensor = tf.random.uniform(shape, dtype=tf.float32)
    return MyTensor(tensor)

