# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32)  # Input shape inferred from X_train.shape (3495,128,128,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model architecture as per provided Sequential model,
        # adjusted to be compatible with TF 2.20.0 and mixed precision handling.
        #
        # Note: The original used mixed_float16 policy, but the last Dense layer
        # was set to dtype float32 explicitly to avoid numerical instability during softmax.
        # We'll do the same here.
        
        self.conv1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(128,128,3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))
        
        self.conv2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        
        # Dense layer forced to float32 dtype to stabilize softmax output when mixed precision enabled
        self.dense2 = tf.keras.layers.Dense(10, dtype='float32')
        
        self.softmax = tf.keras.layers.Activation('softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.softmax(x)
        return out

def my_model_function():
    # Instantiate and return model instance
    return MyModel()

def GetInput():
    # Generate a random tensor matching the expected input: batch_size=2 (chosen per original batch_size)
    # Shape: (2, 128, 128, 3), dtype float32, values in [0,1) to mimic rescaled image input
    return tf.random.uniform((2, 128, 128, 3), dtype=tf.float32)

