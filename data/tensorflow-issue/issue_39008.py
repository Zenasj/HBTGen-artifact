# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from the example training data (batch size B, input size 784)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the "ThreeLayerMLP" model from the issue
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.pred_layer = tf.keras.layers.Dense(10, name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.pred_layer(x)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor shaped (batch_size=64, 784) as training data in the issue used
    # Input dtype is float32 as per typical tf.keras default
    return tf.random.uniform((64, 784), dtype=tf.float32)

