# tf.random.uniform((B, 1), dtype=tf.float32) ← The input is shape (batch_size, 1) since data_x shape is (?,1)

import tensorflow as tf
import numpy as np

# Fixed random seeds for reproducibility
np.random.seed(1)
tf.keras.backend.set_floatx('float32')

# Identity function for baseline
func = lambda x: x

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model expects input shape (?, 1)
        self.func = func
        
        # Layers as per original definition with fixed kernel initializers
        kernel_init = tf.keras.initializers.GlorotUniform(seed=1)
        self.net1 = tf.keras.layers.Dense(
            units=4,
            activation='relu',
            kernel_initializer=kernel_init
        )
        self.net2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=kernel_init
        )
        # Optimizer as per original code
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    def loss_func(self, pred, true):
        # Loss: mean of norm of (func(x)-y) / squared norm of y
        # This modification (square denominator) resolved the nan gradient issue as per issue discussion
        numerator = tf.norm(func(pred) - true, axis=1)
        denominator = tf.math.square(tf.norm(true, axis=1))
        return tf.reduce_mean(numerator / denominator)

    def call(self, x, y):
        # Forward pass: predicted output = x + small correction g = net2(net1(r)) * 2e-3
        # where r = y - func(x)
        r = y - self.func(x)
        g = self.net2(self.net1(r)) * 2e-3
        return x + g

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            y_pred = self(inputs, targets, training=True)
            loss_value = self.loss_func(y_pred, targets)
        grads = tape.gradient(loss_value, self.trainable_variables)
        return loss_value, grads

    def train_one_step(self, x, y):
        loss_value, grads = self.grad(x, y)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value

    def train(self, data_x, data_y, start_time=None, max_iter=3000):
        import time
        if start_time is None:
            start_time = time.time()
        for it in range(max_iter):
            loss_value = self.train_one_step(data_x, data_y)
            print(f"Iter {it:4d}: loss {loss_value.numpy():14.8e},  runtime: {time.time() - start_time:8.2f}")

def my_model_function():
    return MyModel()

def GetInput():
    # The data input shape is (batch_size, 1) as per original data
    # We'll generate a random batch of 64 inputs in the same shape
    # dtype=float32 matches the example
    return tf.random.uniform((64, 1), dtype=tf.float32)

