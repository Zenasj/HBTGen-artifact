# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from "input_shape=(784,)" of first Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the original Sequential model from the reproducer:
        # Dense(512, relu) -> Dropout(0.2) -> Dense(10)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(784,))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel; weights are initialized randomly (no pretrained weights loaded)
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape (batch size arbitrary, here set to 1)
    return tf.random.uniform((1, 784), dtype=tf.float32)

