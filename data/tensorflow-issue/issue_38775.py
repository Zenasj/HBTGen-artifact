# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model demonstrates the behavior discussed in the issue:
    - A custom layer prints tf.executing_eagerly() during the call.
    - When using symbolic inputs (Keras Input), executing_eagerly() inside the layer is False.
    - When using eager tensors as input, executing_eagerly() inside the layer is True.
    
    This mimics the example from the GitHub issue comments.
    """

    class MyLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            # Print eager execution status inside the custom layer call (like in the issue)
            tf.print("tf.executing_eagerly() =", tf.executing_eagerly())
            return inputs

    def __init__(self):
        super().__init__()
        self.my_layer = MyModel.MyLayer()

    def call(self, inputs):
        return self.my_layer(inputs)


def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a symbolic input tensor compatible with the model.
    We choose a batch size of 4 here for demonstration, shape=(4,1).
    
    This matches the input shape expected by MyLayer, e.g., (batch_size, 1).
    """
    # Using symbolic input tensor (Keras Input) emulates the behavior where
    # inside the custom layer, tf.executing_eagerly() will be False.
    return tf.keras.Input(shape=(1,), dtype=tf.float32)

