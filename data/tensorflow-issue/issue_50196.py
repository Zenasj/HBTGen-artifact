# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape from MNIST dataset example used in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mirror the example Keras Sequential model from the issue:
        # Flatten input (28, 28), Dense with 128 units + relu, Dropout 0.2, Dense with 10 output logits
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights to load as per issue; just instantiate.
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (B, 28, 28) mimicking batch of grayscale images.
    # Assumed batch size 32 as a reasonable default for testing.
    batch_size = 32
    # Values normalized to [0,1] float32 as is done in the issue example (MNIST / 255.0)
    return tf.random.uniform((batch_size, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)

