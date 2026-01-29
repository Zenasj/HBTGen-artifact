# tf.random.uniform((B, 1), dtype=tf.float32)  ← inferred input shape is (batch_size, 1) matching Input(shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model matching the example: one input layer
        # No layers beyond input, as the example uses keras.Sequential with only Input layer
        # To have a proper model, add a dummy identity (or linear activation)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        # Forward pass - simple identity mapping with a dense layer
        return self.dense(inputs)


def my_model_function():
    model = MyModel()
    # Compile the model to be ready for training, no loss or optimizer specified in original code
    # Use dummy optimizer and loss for compile to mirror original behavior (model.compile() without args is invalid)
    model.compile(optimizer='adam', loss='mse')
    return model


def GetInput():
    # Return random input tensor shaped (batch_size, 1)
    # Batch size can be a small number, e.g. 5 to match example inputs
    return tf.random.uniform((5, 1), dtype=tf.float32)

