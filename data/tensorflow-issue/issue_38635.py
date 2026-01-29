# tf.random.uniform((100, 31), dtype=tf.int32) ← Inferred input shape from the dataset snippet in the issue (batch size 100, feature size 31)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers same as in the issue's Sequential model, but use Functional/Subclass for clarity
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)  # cast input to float32 as model expects floats
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel for training or inference
    return MyModel()

def GetInput():
    # Return a random tensor input matching the shape and dtype expected by MyModel
    # Based on the original train data: shape (BATCH_SIZE=100, 31 features), int inputs converted to float internally
    # Here using uniform ints in [0,1000) like in the original example, then casting float inside model
    return tf.random.uniform((100, 31), minval=0, maxval=1000, dtype=tf.int32)

