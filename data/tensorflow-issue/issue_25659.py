# tf.random.uniform((B, 100, 1), dtype=tf.float32) and tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shapes

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Branch 1: Conv1D sequence for input shape (100, 1)
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(2, 3, activation='tanh', input_shape=(100, 1)),
            tf.keras.layers.Conv1D(4, 3, activation='tanh'),
            tf.keras.layers.Conv1D(6, 3, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(8, 3, activation='tanh'),
            tf.keras.layers.Conv1D(10, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

        # Branch 2: Dense layers for input shape (1,)
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(1,)),
            tf.keras.layers.Dense(10, activation='linear')
        ])

        # Concatenate layer to merge outputs of branch1 and branch2
        self.concat = tf.keras.layers.Concatenate()

        # Following Dense layers after concatenation
        self.dense1 = tf.keras.layers.Dense(1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')

    def call(self, inputs, training=False):
        # Expect inputs as a list or tuple: [input_for_branch1, input_for_branch2]
        x1, x2 = inputs

        # Pass through each branch
        out1 = self.branch1(x1, training=training)
        out2 = self.branch2(x2, training=training)

        # Concatenate outputs on last dimension
        x = self.concat([out1, out2])

        # Pass through final dense layers
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate example inputs consistent with model input shapes
    # Use batch size of 4 as an example
    batch_size = 4

    # Input for branch1: shape (batch_size, 100, 1)
    input_branch1 = tf.random.uniform((batch_size, 100, 1), dtype=tf.float32)

    # Input for branch2: shape (batch_size, 1)
    input_branch2 = tf.random.uniform((batch_size, 1), dtype=tf.float32)

    return [input_branch1, input_branch2]

