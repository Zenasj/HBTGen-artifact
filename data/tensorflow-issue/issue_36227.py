# tf.random.uniform((1, 2), dtype=tf.float32)  ← Input shape inferred from model.build((1,2)) and input zeros([1,2])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The model reflects the example from the issue:
        # Three layers: Dense(2), Dense(1000), LeakyReLU(alpha=0.1), Dense(500)
        self.layers_list = [
            tf.keras.layers.Dense(2),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(500)
        ]
        
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


def my_model_function():
    # Return an instance of the model fully built
    model = MyModel()
    # Explicitly build with batch size 1 and input dim 2 as in the issue
    model.build((1, 2))
    return model


def GetInput():
    # Return a random float32 tensor shaped (1, 2) matching model input
    # Use tf.random.uniform to match style requested
    return tf.random.uniform((1, 2), dtype=tf.float32)

