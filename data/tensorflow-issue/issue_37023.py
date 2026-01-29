# tf.random.uniform((B=1, H=28, W=28), dtype=tf.float32) ← input shape inferred from MNIST grayscale images (28x28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the "SimpleModel.train" behavior with unrolled GRUCell RNN for MNIST
        self.cell = tf.keras.layers.GRUCell(3)
        self.rnn = tf.keras.layers.RNN(self.cell, unroll=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')

    def call(self, inputs, training=False):
        # inputs shape expected: (batch_size, 28, 28)
        x = self.rnn(inputs)  # output shape: (batch_size, 3)
        x = self.flatten(x)   # flatten in this case is trivial, but kept as in original
        out = self.dense(x)   # output shape: (batch_size, 10)
        return out

def my_model_function():
    # Returns an instance of MyModel with uninitialized weights (initialized on first call)
    return MyModel()

def GetInput():
    # Return a random MNIST-like input: batch size 1, 28 rows, 28 columns, dtype float32
    # The original model input is shaped (28,28) per sample (grayscale image)
    # We'll produce a batch with single example, consistent with the model's input signature
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

