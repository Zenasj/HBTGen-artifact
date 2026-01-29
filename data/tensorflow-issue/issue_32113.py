# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from model.build((None, 10))

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)  # Pass training flag to enable dropout only when training
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel
    # Model can be optionally built with a batch input shape if needed later
    return MyModel()


def GetInput():
    # Return a random input tensor matching the input expected by MyModel: shape = (batch_size, 10)
    # Use float32 dtype for compatibility
    batch_size = 8  # Arbitrary batch size
    return tf.random.uniform((batch_size, 10), dtype=tf.float32)

