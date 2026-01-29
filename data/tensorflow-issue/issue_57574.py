# tf.random.uniform((B, 512), dtype=tf.float32)  ← The input shape is (batch_size, 512)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Add, Input
from tensorflow.keras import Model


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = Dense(512)
        # We use Add layer explicitly because the original addition using `+` breaks serialization.
        self.add = Add()

    def call(self, inputs):
        # Residual connection: shortcut + dense output
        short = inputs
        x = self.dense(inputs)
        x = self.add([short, x])  # Add layer expects a list of tensors
        return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mylayer = MyLayer()

    def call(self, inputs):
        return self.mylayer(inputs)


def my_model_function():
    # Instantiate MyModel.
    # Note: The model must be built (or called once) before saving if used outside Functional API.
    model = MyModel()
    # Build the model with known input shape
    model.build(input_shape=(None, 512))
    return model


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape is (batch_size, 512); batch_size is chosen as 4 (arbitrary)
    return tf.random.uniform((4, 512), dtype=tf.float32)

