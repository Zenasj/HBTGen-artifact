# tf.random.uniform((32, 1), dtype=tf.float32) ← input shape inferred from minimal example (batch 32, feature 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers from the minimal example in issue
        self.dense1 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.activation(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build model to ensure weights are created
    dummy_input = tf.zeros((1, 1), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Return a tensor that matches expected input shape (batch_size=32, features=1)
    return tf.ones((32, 1), dtype=tf.float32)

