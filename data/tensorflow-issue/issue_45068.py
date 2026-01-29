# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from CIFAR-10 dataset (32x32 RGB images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from the provided Sequential model:
        # Conv2D(32, 3x3) + ReLU, MaxPooling2D(2x2)
        # Conv2D(64, 3x3) + ReLU, MaxPooling2D(2x2)
        # Conv2D(64, 3x3) + ReLU
        # Flatten
        # Dense(64) + ReLU
        # Dense(10) for 10 classes
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))
        
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))
        
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, x, training=False):
        x = self.conv1(x)       # (B, 30,30,32)
        x = self.pool1(x)       # (B, 15,15,32)
        x = self.conv2(x)       # (B, 13,13,64)
        x = self.pool2(x)       # (B, 6,6,64)
        x = self.conv3(x)       # (B, 4,4,64)
        x = self.flatten(x)     # (B, 4*4*64=1024)
        x = self.dense1(x)      # (B,64)
        x = self.dense2(x)      # (B,10)
        return x

def my_model_function():
    # Return a new instance of MyModel
    # No special initialization needed beyond default
    return MyModel()

def GetInput():
    # Return a random tensor matching the CIFAR-10 input images shape
    # Use dtype float32, values normalized in [0,1], batch size 1 as example
    batch_size = 1  # can be changed as needed
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0., maxval=1., dtype=tf.float32)

