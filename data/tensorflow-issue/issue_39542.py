# tf.random.uniform((B, 10, 56, 56, 3), dtype=tf.float32)  # Input shape inferred from the reproducer: batch size B, 10 frames, 56x56 RGB images

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load the TF Hub KerasLayer (using the example URL from the issue)
        # This layer processes a single image [56,56,3]
        self.base_model = hub.KerasLayer("https://tfhub.dev/google/bit/s-r101x1/1",
                                         trainable=False)

        # Wrap the base_model in a tf.keras.Model so we can override compute_output_shape
        # Input shape: [batch, 56, 56, 3]; output shape is assumed (batch, 512) as per the user's trick 
        inputs = tf.keras.Input(shape=(56, 56, 3))
        x = self.base_model(inputs)
        self.net = tf.keras.Model(inputs, x)
        # Provide a compute_output_shape override to enable TimeDistributed to work
        self.net.compute_output_shape = lambda input_shape: (input_shape[0], input_shape[1], 512)
        
        # Now wrap net with TimeDistributed to apply on sequences of images
        self.timedist = tf.keras.layers.TimeDistributed(self.net)
        
        # Following layers from the example: LSTM + Dense + Dense
        self.lstm = tf.keras.layers.LSTM(units=512)
        self.dense_relu = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dense_sigmoid = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, time_steps=10, 56, 56, 3)
        x = self.timedist(inputs)  # shape: (batch_size, 10, 512)
        x = self.lstm(x)           # shape: (batch_size, 512)
        x = self.dense_relu(x)     # shape: (batch_size, 128)
        x = self.dense_sigmoid(x)  # shape: (batch_size, 1)
        # Output shape: (batch_size, 1)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape expected by MyModel:
    # shape: (batch_size, 10, 56, 56, 3), dtype float32
    # Use uniform random floats scaled to [0, 255) to mimic image data
    batch_size = 4  # smaller batch size to allow quick testing
    return tf.random.uniform((batch_size, 10, 56, 56, 3), minval=0, maxval=255, dtype=tf.float32)

