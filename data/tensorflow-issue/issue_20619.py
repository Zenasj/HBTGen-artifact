# tf.random.uniform((B, 2)) ← Input shape inferred from issue examples: input features with shape (None, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple feedforward model consistent with examples given:
        # Input shape (batch_size, 2), two Dense layers as in original code snippets
        self.dense1 = tf.keras.layers.Dense(8, activation='relu', input_shape=(2,))
        self.dense2 = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


def my_model_function():
    """
    Returns an instance of MyModel.
    This model is compatible with tf.keras optimizers (recommended over TF native optimizers)
    to work properly with ReduceLROnPlateau callbacks as explained in the issue discussion.
    """
    return MyModel()


def GetInput():
    """
    Returns a random tensor input matching the expected model input shape.
    From the issue, the model input is shape (batch_size, 2).
    We choose batch_size=32 as a common example.
    """
    return tf.random.uniform((32, 2), dtype=tf.float32)

