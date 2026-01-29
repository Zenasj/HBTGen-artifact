# tf.random.uniform((64, 1000, 100), dtype=tf.float32) ← Inferred input shape from the issue: batch_size=64, n_steps=1000, n_input=100

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.n_hidden = 1000
        self.n_input = 100
        
        # Define the RNN and Dense layers as described
        self.rnn = tf.keras.layers.SimpleRNN(
            units=self.n_hidden, 
            return_sequences=True
        )
        self.dense = tf.keras.layers.Dense(units=self.n_input)

    def call(self, inputs, training=False):
        # Forward pass: inputs: shape (batch, time, input_dim)
        x = self.rnn(inputs, training=training)
        x = self.dense(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with SGD optimizer and MSE loss to match original example
    model.compile(optimizer=tf.optimizers.SGD(0.1), loss="mse")
    return model


def GetInput():
    # Return a random tensor consistent with expected input shape and dtype
    # Matching the example's float32 dtype and input shape (batch=64, steps=1000, input=100)
    return tf.random.uniform(shape=(64, 1000, 100), dtype=tf.float32)

