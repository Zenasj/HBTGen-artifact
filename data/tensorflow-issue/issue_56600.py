# tf.random.uniform((B, 8), dtype=tf.float32)  ← Input shape inferred from input_shape=[8] in Dense input layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers exactly as per the sequential model in the issue
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', input_shape=(8,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dense4 = tf.keras.layers.Dense(32, activation='relu')
        self.dense5 = tf.keras.layers.Dense(3)  # Output layer with 3 units, no activation specified (linear)
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected model input shape (batch size 256 here is arbitrary)
    # The original training uses batch_size=256; we mimic that.
    batch_size = 256
    return tf.random.uniform((batch_size, 8), dtype=tf.float32)

