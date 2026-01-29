# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Inferred generic input shape for keras models (batch size B, height H, width W, channels C)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, define a simple CNN model as a submodule
        self.conv = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
        
        # Define a second submodule to demonstrate "fusion" if multiple models were discussed
        # Here just a simple MLP on flattened input for comparison purpose
        self.dense_alt1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_alt2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Apply main CNN pipeline
        x1 = self.conv(inputs)
        x1 = self.flatten(x1)
        output1 = self.dense(x1)
        
        # Apply alternate MLP pipeline (flatten input directly before feeding)
        x2 = self.flatten(inputs)
        x2 = self.dense_alt1(x2)
        output2 = self.dense_alt2(x2)
        
        # Compare outputs - output boolean tensor if outputs are close within tolerance
        close = tf.math.reduce_all(
            tf.math.abs(output1 - output2) < 1e-5, axis=-1, keepdims=True
        )
        
        # Return tuple of outputs and comparison for demonstration
        # Usually real fusion or comparison logic depends on actual use case
        return output1, output2, close

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input of MyModel:
    # Assuming an image batch (B=4) of 28x28 RGB (C=3), dtype float32
    B, H, W, C = 4, 28, 28, 3
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

