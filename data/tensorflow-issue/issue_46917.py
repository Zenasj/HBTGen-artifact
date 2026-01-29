# tf.random.uniform((B, 1), dtype=tf.float32)  ← input is a batch of vectors with shape (B, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the original simple model with two Dense layers:
        self.dense1 = tf.keras.layers.Dense(units=10, activation=None)
        self.dense2 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return a compiled instance of MyModel
    model = MyModel()
    # Build model by calling it once to create weights
    model(tf.zeros([1,1], dtype=tf.float32))
    # Compile as per original example
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def GetInput():
    # Return random inputs matching (batch_size, feature_dim=1)
    # We choose batch size 8 arbitrarily as an example.
    return tf.random.uniform((8, 1), dtype=tf.float32)

