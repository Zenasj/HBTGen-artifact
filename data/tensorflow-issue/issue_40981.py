# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  ← Input shape inferred from Dense layer input_shape=(32,32,3)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the original model with two Dense layers as per the issue example
        self.model_original = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu', name='dense_1', input_shape=(32*32*3,)),
            tf.keras.layers.Dense(units=10, activation='softmax', name='dense_2')
        ])
        # Save config and create a second model from the same config
        config = self.model_original.get_config()
        # Create a new Sequential model from the same configuration
        self.model_from_config = tf.keras.Sequential.from_config(config)
        
    def call(self, inputs):
        # Flatten inputs to match Dense input shape (batch, 32*32*3)
        # because the original Dense layer expects a 1D vector per example
        x = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
        
        # Run both models
        out_original = self.model_original(x)
        out_from_config = self.model_from_config(x)
        
        # Compare configurations - usually not part of model call, so we won't do here
        # Instead, compare weights of first weight matrix (dense_1 kernel)
        w_orig = self.model_original.weights[0]
        w_new = self.model_from_config.weights[0]
        
        # Boolean tensor: element-wise closeness check with tolerance
        close = tf.reduce_all(tf.math.abs(w_orig - w_new) < 1e-5)
        
        return {
            'output_original': out_original,
            'output_new_model': out_from_config,
            'weights_close': close
        }

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor shaped (batch_size, 32, 32, 3) as RGB image-like input
    # Using float32 uniformly sampled values
    batch_size = 2  # arbitrary small batch for testing
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

