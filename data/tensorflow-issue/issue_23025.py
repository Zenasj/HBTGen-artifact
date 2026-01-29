# tf.random.uniform((B, 4), dtype=tf.float32) ← Input shape inferred from model.input_dim=4 in Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # According to the issue code, the model is a Sequential with:
        # Dense(64, relu, input_dim=4) followed by Dense(4, linear)
        # We'll recreate this architecture using functional style inside the subclass.

        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=4, activation='linear')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)


def my_model_function():
    # Return an instance of MyModel, as per instructions.
    return MyModel()


def GetInput():
    # Return a random tensor with shape (B, 4), compatible with the model input
    # We'll choose batch size 2 as in example prediction calls in the issue
    batch_size = 2
    input_dim = 4
    # Use uniform distribution as a reasonable default matching dtype float32
    return tf.random.uniform(shape=(batch_size, input_dim), dtype=tf.float32)

