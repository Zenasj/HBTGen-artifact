# tf.random.uniform((B, T, F), dtype=tf.float32) ← Input shape inferred as (batch_size, time_steps, features), e.g. (1, 96, 100)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A simple subclass model wrapping a single LSTM recurrent layer with use_bias=True.
    This model reproduces the scenario described in the issue where LSTM with use_bias=True
    can cause a ValueError on some Linux platforms during model.fit(),
    while explicit calls work fine.

    The forward pass just invokes the LSTM layer on the input.
    """

    def __init__(self):
        super().__init__()
        # The risky configuration per issue: use_bias=True (default)
        self.lstm = tf.keras.layers.LSTM(64, use_bias=True)

    def call(self, inputs, training=False):
        return self.lstm(inputs, training=training)


def my_model_function():
    """
    Returns an instance of MyModel.

    The user can create the model and then compile or fit as desired.

    This mirrors the minimal reproducible model snippet from the issue:
    class FailModel(Model):
        def __init__(self):
            super().__init__()
            self.lstm = layers.LSTM(64, use_bias=True)

        def call(self, input, training=False):
            return self.lstm(input)
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected input to MyModel.

    Assumptions:
    - batch size: 1 (can be arbitrary)
    - time steps: 96 (from issue example input)
    - features: 100 (typical input feature dim from example)
    - dtype: float32
    """
    batch_size = 1
    time_steps = 96
    features = 100
    return tf.random.uniform((batch_size, time_steps, features), dtype=tf.float32)

