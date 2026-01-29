# tf.random.uniform((1, 256, 64, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the failing pattern:
        # Input shape: (batch_size=1, 256, 64, 3)
        # Reshape to (16384, 3), Softmax over axis=1
        # Flatten and then Dense(1)
        # Note: batch_size fixed to 1 to avoid dynamic shape issues flagged in the issue.
        
        self.reshape = tf.keras.layers.Reshape((16384, 3))
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.softmax(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Note: model expects input shape (1, 256, 64, 3),
    # fixed batch size 1 to avoid dynamic size issues in TFLite GPU delegate.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input shape expected by MyModel.
    # Batch size fixed to 1 as per recommendations in the issue.
    return tf.random.uniform((1, 256, 64, 3), dtype=tf.float32)

