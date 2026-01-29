# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Inferred input shape from the issue's keras Input layer examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the input and output Dense layer as described in the issue
        # The user example builds model with Input(shape=(224,224,3)), Dense(10, dtype=tf.float32 or float16)
        # We use tf.float32 internally because tf.float16 led to TFLiteConverter errors in TF 2.2.0
        # Note: Users had issues with float16 dtype directly on input or Dense layer
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer producing 10 output units, using float32 (safe for TFLite conversion)
        self.dense = tf.keras.layers.Dense(10, dtype=tf.float32)  
        
    def call(self, inputs):
        x = self.flatten(inputs)  # Flatten spatial dimensions before Dense
        out = self.dense(x)
        return out

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching input expected by MyModel: 
    # shape (1, 224, 224, 3), dtype float32 (float16 input leads to conversion error in TF 2.2)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

