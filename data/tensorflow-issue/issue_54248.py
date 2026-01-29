# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset preprocessing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model according to the functional create_keras_model function logic
        self.reshape = tf.keras.layers.Reshape((28 * 28 * 1,), name="Input_Reshape")
        self.dense1 = tf.keras.layers.Dense(32, name="Layer_0")
        self.dense2 = tf.keras.layers.Dense(32, name="Layer_1")
        self.dense3 = tf.keras.layers.Dense(32, name="Layer_2")
        self.output_layer = tf.keras.layers.Dense(10, name="Output_Layer", activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        out = self.output_layer(x)
        return out

def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor matching the input shape used in the original example
    # Batch size can be arbitrary; using 32 like in training/testing batch size
    batch_size = 32
    # MNIST images are grayscale 28x28
    input_tensor = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    return input_tensor

