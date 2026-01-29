# tf.random.uniform((B, 180, 320*4, 3), dtype=tf.float32)  # inferred input shape from issue: images are 180 height, 320*4 width, 3 channels

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreating the model architecture described in the issue:
        # - Conv2D(4, 9x9, relu), maxpool 2x2
        # - Conv2D(8, 9x9, relu), maxpool 2x2
        # - Conv2D(16, 9x9, relu), maxpool 2x2
        # - Flatten + Dense(5, sigmoid)
        # Using tf.keras.Sequential inside the model for clarity.
        
        self.features = tf.keras.Sequential([
            layers.Conv2D(4, (9, 9), activation='relu', padding='valid', input_shape=(180, 320*4, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (9, 9), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (9, 9), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(5, activation='sigmoid')
        ])
    
    def call(self, inputs, training=False):
        return self.features(inputs)

def my_model_function():
    # Instantiate and return the model instance
    model = MyModel()
    # Normally you would compile the model outside, but since not requested, just return it.
    return model

def GetInput():
    # Return random tensor matching the input expected by MyModel:
    # Batch size variable, but let's pick batch size 4 as an example.
    # Input shape (batch_size, height=180, width=320*4=1280, channels=3)
    batch_size = 4
    input_tensor = tf.random.uniform(shape=(batch_size, 180, 320*4, 3), dtype=tf.float32)
    return input_tensor

