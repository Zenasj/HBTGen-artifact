# tf.random.uniform((16, 128, 128, 1), dtype=tf.float32) ← Batch size 16, image height 128, width 128, single channel grayscale input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the CNN model described in the issue with the fix using Activation('relu') 
        # instead of the problematic ReLU() layer class that caused loading errors in TF 1.11.
        # Using the functional pattern inside the custom model class.
        
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation='linear', kernel_initializer='he_uniform')
        # Use Activation('relu') layer explicitly to avoid serialization issue:
        self.relu1 = tf.keras.layers.Activation('relu')
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5))
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.relu2 = tf.keras.layers.Activation('relu')
        self.predictions = tf.keras.layers.Dense(4, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pooling1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.relu2(x)
        # Note: The original corrected post mistakenly re-flattened conv1 output before final Dense,
        # that seems an error (no reason to flatten conv1 again). Using flattened dense output for final layer.
        out = self.predictions(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected shape:
    # batch_size=16, height=128, width=128, channels=1, dtype float32
    # Using tf.random.uniform for tensor input generation as per requirement.
    return tf.random.uniform((16, 128, 128, 1), dtype=tf.float32)

