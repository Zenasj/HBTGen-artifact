# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from the original keras model input_shape=(10,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the original simple keras.Sequential model given in chunk 1:
        # Single Dense layer with 1 unit, sigmoid activation, input_shape=(10,)
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return x

def my_model_function():
    # Instantiate and compile the model similarly to the original code snippet
    model = MyModel()
    # Compile with same parameters as original snippet: binary_crossentropy loss, sgd optimizer
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model

def GetInput():
    # Return a random tensor input matching the expected input for MyModel
    # Input shape is (batch_size, 10), batch_size is arbitrarily chosen as 8 here
    return tf.random.uniform((8, 10), dtype=tf.float32)

