# tf.random.uniform((B, 2), dtype=tf.float64)  ← Input shape inferred as batch_size x 2 (x, t)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two hidden layers with 100 units each and ReLU activation, final layer linear as per original code
        self.dense1 = tf.keras.layers.Dense(100, activation='relu', dtype=tf.float64)
        self.dense2 = tf.keras.layers.Dense(100, activation='relu', dtype=tf.float64)
        self.out_layer = tf.keras.layers.Dense(1, activation='linear', dtype=tf.float64)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.out_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel initialized with appropriate dtype float64
    return MyModel()

def GetInput():
    # Generate a random input tensor shaped (batch_size, 2), dtype float64, matching x and t inputs
    # Assume batch size of 500 like in original batching
    batch_size = 500
    # x ∈ [-pi, pi], t ∈ [0, 2] from original parameter scan boundaries
    x = tf.random.uniform((batch_size, 1), minval=-3.14159265359, maxval=3.14159265359, dtype=tf.float64)
    t = tf.random.uniform((batch_size, 1), minval=0.0, maxval=2.0, dtype=tf.float64)
    xt = tf.concat([x, t], axis=1)
    return xt

