# tf.random.uniform((B, 512, 1), dtype=tf.float32) ← inferred input shape from issue descriptions and data arrays

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model takes sequences of shape (None, 1) with length 512 inferred
        # Based on original model architecture given:
        # Conv1D -> BiGRU (128) -> BiGRU (256) -> Dense (128) -> two output heads
        
        # Layers defined similarly to the original model snippet:
        self.conv1d = tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding="causal",
            activation="relu",
            input_shape=(None, 1)
        )
        
        self.bigru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, activation="tanh", return_sequences=True)
        )
        self.bigru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(256, activation="tanh", return_sequences=True)
        )
        self.dense128 = tf.keras.layers.Dense(128, activation="tanh")
        
        # Output heads: ed (linear) and sd (sigmoid)
        self.dense_ed = tf.keras.layers.Dense(1, activation="linear", name="ed")
        self.dense_sd = tf.keras.layers.Dense(1, activation="sigmoid", name="sd")
    
    def call(self, inputs, training=False):
        """
        Forward pass inputs through the model, returning a dictionary of outputs.
        Equivalent to Keras functional outputs: [o1, o2]
        
        inputs: Tensor of shape (batch, 512, 1)
        returns: dict with keys 'ed' and 'sd' with shapes (batch, 512, 1)
        """
        x = self.conv1d(inputs)
        x = self.bigru1(x)
        x = self.bigru2(x)
        x = self.dense128(x)
        ed_output = self.dense_ed(x)
        sd_output = self.dense_sd(x)
        # Return in dictionary naming output to be compatible with original model
        return {"ed": ed_output, "sd": sd_output}

def my_model_function():
    """
    Returns an instance of MyModel matching the original model architecture,
    ready to be compiled and trained.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor matching the expected input shape.
    
    According to the issue, inputs have shape (batch_size, 512, 1).
    Batch size is interpreted as a flexible dimension; choose an example batch of size 2.
    """
    batch_size = 2  # example batch size
    sequence_length = 512  # fixed from dataset shape
    channels = 1
    return tf.random.uniform((batch_size, sequence_length, channels), dtype=tf.float32)

