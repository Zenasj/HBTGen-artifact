# tf.random.uniform((B,), dtype=tf.float32) ← Input is a 1D tensor representing a batch of scalar float values

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple linear model matching: Dense layer with 1 unit and input shape [1]
        self.dense = tf.keras.layers.Dense(units=1, input_shape=[1])

    def call(self, inputs):
        # Expect inputs shape: (batch_size,)
        # Reshape inputs to (batch_size, 1) to match Dense input requirements
        x = tf.expand_dims(inputs, axis=-1)
        return self.dense(x)


def my_model_function():
    # Create an instance of MyModel
    model = MyModel()

    # To match the example script, compile model and train briefly to initialize weights
    # (Weights are required to avoid errors in usage)
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Use analogous training data as in the issue example
    xs = tf.constant([-1., 0., 1., 2., 3., 4.], dtype=tf.float32)
    ys = tf.constant([-3., -1., 1., 3., 5., 7.], dtype=tf.float32)

    # Fit model for 500 epochs to replicate original training
    model.fit(xs, ys, epochs=500, verbose=0)

    return model


def GetInput():
    # Return a batch of random input floats shaped (batch_size,)
    # Using batch size 4 here as an example
    batch_size = 4
    # Random uniform input between -10 and 10, dtype float32
    return tf.random.uniform((batch_size,), minval=-10.0, maxval=10.0, dtype=tf.float32)

