# tf.random.uniform((B, 50, 50, 3), dtype=tf.float32) ← Input tensor shape inferred from example with images of shape (50, 50, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the original conv BN ReLU stack from the issue example
        self.conv_layers = []
        for _ in range(3):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid'))
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_layers.append(tf.keras.layers.ReLU())
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2)
        
        # The original issue involved TPU model wrapping and an attribute error related to checkpoint saved epoch.
        # Since the issue was about TPU keras_to_tpu_model compatibility, here we recreate a single Keras model 
        # that you could wrap for TPU separately. The TPU-specific wrapping is not included as contrib.tpu is deprecated.
        #
        # This class provides just the functional model core that was used in the reported TensorFlow 1.x bug.
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            # For BatchNormalization, pass training flag correctly
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def my_model_function():
    # Instantiate and return the model.
    # Weight initialization matches default Keras initializers.
    return MyModel()


def GetInput():
    # Generate a random tensor simulating a batch of 128 images with height=50, width=50, channels=3
    # Matches the original example input shape from TensorFlow 1.x TPU bug report
    batch_size = 128
    height = 50
    width = 50
    channels = 3
    # Use float32 inputs as typical for image data in TF/Keras
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

