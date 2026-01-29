# tf.random.uniform((None, 4), dtype=tf.float32) ← Input shape inferred: iris dataset has 4 features per sample

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct a model similar to the one described in the issue (Sequential with Dense layers)
        # Input shape is (4,) as per iris data features
        self.dense1 = tf.keras.layers.Dense(512, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(512 // 2, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(512 // 4, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(512 // 8, activation='tanh')
        self.dense5 = tf.keras.layers.Dense(32, activation='relu')
        self.dense6 = tf.keras.layers.Dense(3, activation='softmax')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor simulating a batch of iris data input samples
    # Iris dataset has 4 features per sample
    # Batch size is set arbitrarily (e.g., 8 samples)
    batch_size = 8
    input_shape = (batch_size, 4)
    # Using float32 as typical dtype for deep learning inputs
    return tf.random.uniform(input_shape, dtype=tf.float32)

