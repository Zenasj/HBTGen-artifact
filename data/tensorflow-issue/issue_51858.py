# tf.random.uniform((1, 3, 3, 32), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Three Conv2D layers with outputs channels: 2, 16, 32 
        # Each with kernel size (1,1) and ReLU activation
        # Input shape per conv layer is (None, 3, 3, 32)
        self.conv2d_2 = tf.keras.layers.Conv2D(2, (1,1), activation='relu')
        self.conv2d_16 = tf.keras.layers.Conv2D(16, (1,1), activation='relu')
        self.conv2d_32 = tf.keras.layers.Conv2D(32, (1,1), activation='relu')

    def call(self, inputs):
        # Forward pass producing three outputs
        o1 = self.conv2d_2(inputs)   # expected shape (batch, 3, 3, 2)
        o2 = self.conv2d_16(inputs)  # expected shape (batch, 3, 3, 16)
        o3 = self.conv2d_32(inputs)  # expected shape (batch, 3, 3, 32)
        
        # Because the original issue described output order changing after TFLite conversion,
        # we'll produce outputs explicitly in order consistent with model definition:
        # (o1, o2, o3)
        
        # No complex fusion or comparison logic given - 
        # the core issue is preserving output order.
        return [o1, o2, o3]

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input shape:
    # batch_size=1, height=3, width=3, channels=32, dtype float32
    return tf.random.uniform((1, 3, 3, 32), dtype=tf.float32)

