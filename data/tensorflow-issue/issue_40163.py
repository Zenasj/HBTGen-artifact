# tf.random.uniform((None, 2), dtype=tf.float32) ← Input shape is (?, 2) corresponding to concatenated x and y coordinates

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers configuration matching original: layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.layers_sizes = [20, 20, 20, 20, 20, 20, 20, 20]

        # Xavier initialization helper
        def xavier_init(in_dim, out_dim):
            stddev = np.sqrt(2/(in_dim + out_dim))
            return tf.random.truncated_normal([in_dim, out_dim], stddev=stddev)

        # Input layer size is 2 (x, y)
        in_dim = 2
        self.weights = []
        self.biases = []
        for out_dim in self.layers_sizes:
            w = tf.Variable(xavier_init(in_dim, out_dim), trainable=True, dtype=tf.float32)
            b = tf.Variable(tf.zeros([out_dim], dtype=tf.float32), trainable=True)
            self.weights.append(w)
            self.biases.append(b)
            in_dim = out_dim
        # Output layer weights and bias (from last 20 to 1 output)
        w_out = tf.Variable(xavier_init(in_dim, 1), trainable=True, dtype=tf.float32)
        b_out = tf.Variable(tf.zeros([1], dtype=tf.float32), trainable=True)
        self.weights.append(w_out)
        self.biases.append(b_out)

        # Store lb, ub placeholders to normalize inputs
        # As the original code does normalization based on data min/max,
        # here we will fix lb=0.0 and ub=1.0 assuming the input is normalized or scaled.
        # This replicates:
        # H = 2.0*(X - lb)/(ub - lb) - 1.0  # scales input to [-1,1]
        self.lb = tf.constant([0.0, 0.0], dtype=tf.float32)
        self.ub = tf.constant([1.0, 1.0], dtype=tf.float32)

    def call(self, inputs):
        # Inputs shape: (batch_size, 2)
        # Normalize inputs to [-1, 1]
        H = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

        # Forward through hidden layers with tanh activation
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, w) + b)
        # Output layer (linear activation)
        out = tf.matmul(H, self.weights[-1]) + self.biases[-1]

        # Output shape: (batch_size, 1)
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor of shape (batch_size, 2)
    # Use batch size = 64 as reasonable default batch size
    # Inputs lie in [0,1] for each coordinate to match normalization assumptions.
    batch_size = 64
    return tf.random.uniform(shape=(batch_size, 2), minval=0.0, maxval=1.0, dtype=tf.float32)

