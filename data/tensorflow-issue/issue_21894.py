# tf.random.uniform((B, 4, 1), dtype=tf.float32) ← inferred input shape (batch, input_size=4, input_features=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_size=2):
        super().__init__()
        # A simple Dense layer mapping the inputs to output_size features
        # This corresponds to the user's example where Dense(OUTPUT_SIZE) is used
        self.dense = tf.keras.layers.Dense(output_size, activation=None)

    def call(self, inputs, training=False):
        # inputs expected shape: (batch, input_size=4, input_features=1)
        # Flatten the time dimension and features for Dense layer input:
        # Shape (batch, 4, 1) -> (batch, 4*1=4)
        x = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default OUTPUT_SIZE=2
    return MyModel()

def GetInput():
    # Generate a random tensor matching input shape (batch, 4, 1)
    # Batch size is assumed dynamic, so pick a small default batch, e.g. 8
    batch_size = 8
    input_size = 4
    input_features = 1
    return tf.random.uniform((batch_size, input_size, input_features), dtype=tf.float32)

