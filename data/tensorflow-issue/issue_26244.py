# tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32) ← inferred input shape from CIFAR-10 dataset example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN similar to the example: Conv2D(32, kernel=3x3) + Flatten + Dense(10)
        # This references the CIFAR-10 input shape: (32, 32, 3)
        self.conv = tf.keras.layers.Conv2D(32, (3, 3), activation=None)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    """
    Initializes and returns an instance of MyModel.
    Following best practice, model initialization happens without explicit device 
    placement here; device setting is left to the user environment if needed.
    """
    model = MyModel()
    # Compile the model with RMSprop optimizer as per original code snippet
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    """
    Returns a random input tensor matching the expected input shape for MyModel.
    Mimics the normalized CIFAR-10 input shape: (batch_size, 32, 32, 3), dtype float32.
    Using a moderate batch size (e.g., 64) for demonstration.
    """
    batch_size = 64
    return tf.random.uniform(shape=(batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

