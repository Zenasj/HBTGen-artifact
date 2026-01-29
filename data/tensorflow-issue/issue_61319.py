# tf.random.uniform((B, 8), dtype=tf.float32) ← Based on input_dim=8 from issue description

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the simple dense neural network given in the issue:
        # Sequential model layers:
        # normalizer (not explicitly provided, so we assume identity here)
        # Dense(512 relu), Dense(512 relu), Dense(1 sigmoid)
        # Input dim: 8 as per user's original model input
        
        # Since normalizer isn't provided, we use Identity Lambda layer (pass-through).
        self.normalizer = layers.Lambda(lambda x: x, name="normalizer_identity")
        
        self.dense1 = layers.Dense(512, activation='relu', name="dense_1")
        self.dense2 = layers.Dense(512, activation='relu', name="dense_2")
        self.dense3 = layers.Dense(1, activation='sigmoid', name="dense_3")
    
    def call(self, inputs):
        x = self.normalizer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    # Note: No special initialization or pre-loaded weights since none provided
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input of shape (B, 8)
    # We choose batch size 1 for convenience
    # dtype float32 since the original model was float32 input
    # Input values between 0 and 1 as a reasonable assumption
    return tf.random.uniform(shape=(1, 8), minval=0.0, maxval=1.0, dtype=tf.float32)

