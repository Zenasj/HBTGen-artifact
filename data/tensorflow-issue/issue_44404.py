# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from original model input_shape=(28, 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as per the tutorial model
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28), dtype=tf.float32)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        # Enforce inputs to be float32 and check shape to avoid matrix size errors
        # The original error was due to unexpected input shape (28, 28, 2) instead of (28, 28).
        # So we squeeze extra channels if present or convert input to expected shape.
        x = tf.convert_to_tensor(inputs)
        
        # If input has 3 dimensions (e.g. (batch, 28, 28, channels)) and channels=2,
        # reduce to one channel by averaging - inferred fix.
        if len(x.shape) == 4 and x.shape[-1] == 2:
            x = tf.reduce_mean(x, axis=-1)
        
        # If input rank is 3 (batch, 28, 28, or batch, 28, 28 with no channel dim), it's fine.
        # Just ensure dtype float32 for compatibility.
        x = tf.cast(x, tf.float32)
        
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching input expected by MyModel:
    # Shape: (batch_size=1, 28, 28)
    # dtype: float32
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

