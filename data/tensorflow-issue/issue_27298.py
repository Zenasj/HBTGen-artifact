# tf.random.uniform((B, 2), dtype=tf.float32)  # Input assumed as batch size B, feature dim 2, inferred from issue examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_labels=10):
        super().__init__()
        # Attempting to reflect the discussion on name scopes in Keras functional API:
        # Using submodules with explicit tf.name_scope inside call, which works during eager execution.
        self.dense1 = tf.keras.layers.Dense(4, name="dense_shitty")
        self.dense2 = tf.keras.layers.Dense(output_labels, name="dense_wok")

    def call(self, x):
        # Wrap layers calls in tf.name_scope to show scopes in eager mode (as done in the "Douche" class example)
        with tf.name_scope("Turd"):
            x = self.dense1(x)
        with tf.name_scope("Sandwich"):
            x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default output_labels=10 as per original keras.layers.Dense(10) examples
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input of MyModel
    # Based on shapes from issue examples: shape [batch_size, 2], batch_size arbitrarily chosen as 8.
    return tf.random.uniform((8, 2), dtype=tf.float32)

