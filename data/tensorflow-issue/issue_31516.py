# tf.random.uniform((100, 10, 6), dtype=tf.float64)  ← Inferred input shape: (batch_size=100, d=10, time_steps=6 (n_time+1))

import tensorflow as tf
import numpy as np
from tensorflow import keras
from scipy.stats import multivariate_normal as normal

d = 10
T = 0.1
n_time = 5
n_sample = 100

def f_tf(t, X, Y, Z):
    # Function from issue, unused in model but included for completeness
    V = Y - tf.math.sin(Y)
    return V

def g_tf(t, X):
    # Output function - reduces sum of cubes along d-axis, keep dims for shape compatibility
    V = tf.math.reduce_sum(X ** 3, axis=1, keepdims=True)
    return V

def k_tf(n_sample):
    # Sample generation function returning Brownian increments W and cumulative X_sample paths
    W = np.zeros([n_sample, d, n_time], dtype=np.float64)
    X_sample = np.zeros([n_sample, d, n_time + 1], dtype=np.float64)
    for i in range(n_time):
        # Each time sample a normal vector per sample (mean=0, cov=1)
        W[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(d, dtype=np.float64), cov=1, size=n_sample), (n_sample, d))
        X_sample[:, :, i + 1] = W[:, :, i]
    return W, X_sample

def nn_tf(x):
    # Simple neural network block using BatchNormalization with dtype tf.float64 as per fix
    x = keras.layers.BatchNormalization(batch_size=n_sample, dtype=tf.float64)(x)
    x = keras.layers.Dense(d, batch_size=n_sample)(x)
    x = keras.layers.BatchNormalization(batch_size=n_sample, dtype=tf.float64)(x)
    return x

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers reused inside nn_tf logic, we instantiate here to keep track
        self.bn1 = keras.layers.BatchNormalization(batch_size=n_sample, dtype=tf.float64)
        self.dense = keras.layers.Dense(d, batch_size=n_sample)
        self.bn2 = keras.layers.BatchNormalization(batch_size=n_sample, dtype=tf.float64)

    def nn_block(self, x):
        x = self.bn1(x)
        x = self.dense(x)
        x = self.bn2(x)
        return x

    def call(self, inputs):
        # Inputs: tuple or list of two tensors: X with shape (n_sample, d, n_time+1), dW with shape (n_sample, d, n_time)
        X, dW = inputs
        batch = tf.shape(X)[0]

        # Initialize outputs Y and Z
        Y = tf.zeros([batch, 1], dtype=tf.float64)
        Z = tf.zeros([batch, d], dtype=tf.float64)

        # Iterate through time steps (n_time-1)
        for it in range(n_time - 1):
            # Accumulate Y with inner product of Z and dW slice at time it
            Y += tf.math.reduce_sum(Z * dW[:, :, it], axis=1, keepdims=True)

            # Reshape X slice at time it for feeding into nn_block
            subX = tf.reshape(X[:, :, it], shape=[batch, d])
            Z = self.nn_block(subX) / d

        # Final Y update for last time step
        Y += tf.math.reduce_sum(Z * dW[:, :, n_time - 1], axis=1, keepdims=True)
        return Y

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate input tensors matching the expected input of MyModel:
    # - X shape: (n_sample, d, n_time+1) with dtype float64
    # - dW shape: (n_sample, d, n_time) with dtype float64
    dW_np, X_np = k_tf(n_sample)
    # Convert numpy arrays to tf.Tensor of dtype tf.float64
    dW_tensor = tf.convert_to_tensor(dW_np, dtype=tf.float64)
    X_tensor = tf.convert_to_tensor(X_np, dtype=tf.float64)
    return (X_tensor, dW_tensor)

