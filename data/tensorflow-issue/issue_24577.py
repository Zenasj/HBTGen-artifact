# tf.random.uniform((B, 2), dtype=tf.float32) ← Based on input shape (batch_size, 2) from the dataset slices

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer producing 1 output as per original Sequential model
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate MyModel and compile similarly to the original example
    model = MyModel()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='mse')
    return model

def GetInput():
    # Return a batch of inputs matching shape (batch_size, 2)
    # Using batch_size=32 to match original example's batch size
    batch_size = 32
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)

