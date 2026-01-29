# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Inferred CIFAR-10 input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating the original Sequential model architecture described:
        # Input: (32, 32, 3)
        # Conv2D with 16 filters, kernel size (3,3)
        # Flatten
        # Dense with 10 units, softmax activation
        
        self.conv2d = tf.keras.layers.Conv2D(16, (3, 3))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Returns an instance of the model initialized with random weights
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (1, 32, 32, 3) with float32 values scaled [0,1]
    # This matches expected CIFAR-10 input shape for the model.
    return tf.random.uniform((1, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

