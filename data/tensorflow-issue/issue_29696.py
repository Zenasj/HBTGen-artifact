# tf.random.uniform((3, 20, 3), dtype=tf.float32) ← Inferred input shape from batch_input_shape in SimpleRNN

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        uniform_regularizer = 0.0
        # SimpleRNN with 5 units, no bias, he_normal initializers, l1 regularizer (0 here)
        self.rnn = tf.keras.layers.SimpleRNN(
            units=5,
            kernel_initializer='he_normal',
            recurrent_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l1(l=uniform_regularizer),
            use_bias=False,
            stateful=True,
            batch_input_shape=(3, 20, 3)  # batch=3, timesteps=20, features=3
        )
        # Dense layer with 3 units (output dim), no bias, he_normal initializer and l1 reg.
        self.dense = tf.keras.layers.Dense(
            3,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l1(l=uniform_regularizer),
            use_bias=False
        )

    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return random input with shape (3, 20, 3), dtype float32 to match SimpleRNN input
    return tf.random.uniform((3, 20, 3), dtype=tf.float32)

