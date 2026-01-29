# tf.random.uniform((B, 28, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers roughly matching the MNIST example discussed
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)  # logits for 10 classes
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization.
    # Matches the small MNIST classifier that was used in the issue discussion.
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped as MNIST images:
    # (batch size, height, width)
    # Assumption: batch size 32 for reasonable default input
    batch_size = 32
    height, width = 28, 28
    # Values normalized to [0,1], matching input preprocessing of the MNIST data
    return tf.random.uniform((batch_size, height, width), minval=0.0, maxval=1.0, dtype=tf.float32)

