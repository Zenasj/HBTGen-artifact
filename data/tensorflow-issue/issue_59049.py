# tf.random.uniform((B, 112, 112, 3), dtype=tf.float32)  # input shape inferred from the issue Input layer

import tensorflow as tf
from tensorflow.keras import layers

class CNNExample(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple CNN example similar to what might be expected for input shape (112, 112, 3).
        # This is inferred since original cnn_example() is not provided.
        self.conv1 = layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool = layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, n_repeats=3):
        """
        This model stacks n_repeats instances of CNNExample horizontally,
        then aggregates their outputs with a GlobalAveragePooling1D layer and a final Dense output.
        
        This replicates the horizontally stacked layers pattern from the issue.
        
        - n_repeats: number of parallel CNN branches
        """
        super().__init__()
        self.n_repeats = n_repeats
        # Create multiple CNNExample sub-models
        self.cnn_branches = [CNNExample() for _ in range(n_repeats)]
        # The lambda will stack outputs along a new axis (1)
        self.stack_layer = layers.Lambda(lambda x: tf.stack(x, axis=1))
        self.global_pool_1d = layers.GlobalAveragePooling1D()
        self.final_dense = layers.Dense(1)

    def call(self, inputs):
        # Run all branches on the same input independently (horizontal stacking)
        outputs = [cnn_branch(inputs) for cnn_branch in self.cnn_branches]
        # Stack along axis 1 to create shape [batch, n_repeats, features]
        stacked = self.stack_layer(outputs)
        # Average pooling over the stacked dimension
        pooled = self.global_pool_1d(stacked)
        # Final prediction
        out = self.final_dense(pooled)
        return out

def my_model_function():
    # Return an instance of MyModel with default 3 repeats.
    # This can be changed to any positive integer as needed.
    return MyModel(n_repeats=3)

def GetInput():
    # Return a random input tensor with shape (batch_size, 112, 112, 3)
    # Use batch size = 4 as a reasonable default for demonstration.
    batch_size = 4
    return tf.random.uniform((batch_size, 112, 112, 3), dtype=tf.float32)

