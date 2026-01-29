# tf.random.uniform((100, 28, 28), dtype=tf.float32) ← Inferred input shape from batch_size=100 and fashion_mnist images 28x28 grayscale

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input 28x28 to 784
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Large Dense layer (reduced from 2048*2048 to 2048 to prevent RAM exhaustion)
        # Original: Dense(2048*2048) is ~3.3 billion params, causes OOM
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        # Output layer for 10 classes
        self.dense2 = tf.keras.layers.Dense(10)
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel with weights initialized randomly
    return MyModel()

def GetInput():
    # Return a random tensor matching the shape and dtype of typical fashion_mnist input
    # Batch size 100 (as per original training batch_size)
    # Input shape: (28, 28), grayscale images, float32
    return tf.random.uniform((100, 28, 28), dtype=tf.float32)

