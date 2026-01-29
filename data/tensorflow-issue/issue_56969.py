# tf.random.uniform((4092, 2), dtype=tf.float32) ← Input shape based on X_data with shape (N_0, 2)

import tensorflow as tf

lb = tf.constant([0, 0], dtype=tf.float32)  # lower bound for input features
ub = tf.constant([1, 1], dtype=tf.float32)  # upper bound for input features

def fun_u_0(xx):
    """
    Target function: returns 1.0 if input is within a circle centered at midpoint with radius rr,
    otherwise 0.0. This shapes a circular label region.
    """
    c_0 = 0.5 * (lb + ub)
    rr = 0.25 * tf.reduce_min(ub - lb)
    dsq = tf.reduce_sum((xx - c_0) ** 2, axis=1)
    return tf.where(dsq <= rr * rr, 1.0, 0.0)

class MyModel(tf.keras.Model):
    """
    A small fully connected network, configurable for number of hidden layers and neurons.
    Uses LeakyReLU activations, 'glorot_uniform' initializer.
    Output is a scalar per sample.
    """

    def __init__(self, num_hidden_layers=2, num_neurons_per_layer=64):
        super().__init__()
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    num_neurons_per_layer,
                    activation=tf.keras.layers.LeakyReLU(),
                    kernel_initializer="glorot_uniform",
                )
            )
        self.output_layer = tf.keras.layers.Dense(
            1, kernel_initializer="glorot_uniform"
        )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        return out


def my_model_function():
    # Return an instance of MyModel with 2 hidden layers and 64 neurons each, matching the original example
    return MyModel(num_hidden_layers=2, num_neurons_per_layer=64)


def GetInput():
    """
    Generate a batch of random inputs matching the model's expected input shape.
    Shape: (4092, 2), float32, uniform between lb and ub.
    """
    N_0 = 4092
    return tf.random.uniform((N_0, 2), lb, ub, dtype=tf.float32)

