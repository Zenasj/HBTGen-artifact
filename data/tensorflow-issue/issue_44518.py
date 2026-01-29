# tf.random.uniform((5, 1), dtype=tf.float32) ← Input shape inferred from example inputs in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer model as in the issue example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)


def my_model_function():
    # Returns a compiled instance of MyModel with Adam optimizer and MSE loss,
    # matching the prepare_model() function in the issue example.
    model = MyModel()
    model.compile(optimizer='adam', loss='mse')
    model.build([None, 1])  # Input shape consistent with example
    return model


def GetInput():
    # Input shape (5, 1) matches the example inputs used for training in the issue.
    # Using uniform distribution as per example inputs.
    return tf.random.uniform((5, 1), dtype=tf.float32)

