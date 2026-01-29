# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape based on CIFAR-10 dataset used in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CNN model as per "Code 2" in the issue
        
        # Convolutional + MaxPooling layers
        self.conv = tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        
        # Flatten and Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)  # Output logits for 10 classes
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits

def my_model_function():
    # Return an instance of the model
    # The model is structured as in the crashing example (Code 2) from the issue so that input/output shapes
    # and layer details match.
    return MyModel()

def GetInput():
    # Return a single input tensor matching the shape expected by the model: batch size can be any positive int.
    # Using batch size 1 as per the crashing command in the issue.
    batch_size = 1
    # CIFAR-10 images are 32x32 with 3 channels, float32 normalized between 0 and 1
    return tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)

