# tf.random.uniform((32, 60, 2), dtype=tf.float32) ← inferred input shape and dtype from issue (batch_size=32, features=(60,2))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example from the issue:
        # Input shape: (60, 2)
        # Simple model with two Dense layers
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')
        
    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, 60, 2)
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

def my_model_function():
    # Create and return an instance of MyModel
    model = MyModel()
    # Optional: build the model by calling it once with input shape (None, 60, 2)
    dummy_input = tf.zeros((1, 60, 2), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor input matching model input shape (batch_size=32, 60, 2)
    # Using float32 dtype as typical for TensorFlow models
    batch_size = 32
    sequence_length = 60
    feature_dim = 2
    return tf.random.uniform(shape=(batch_size, sequence_length, feature_dim), dtype=tf.float32)

