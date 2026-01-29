# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) ← Based on input shape in my_model function (im_height=64, im_width=64, channels=3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the feature extractor model as described in the issue (my_model)
        # This model includes Conv2D, BatchNormalization, ReLU, MaxPool, Conv2D, BatchNormalization, ReLU
        self.feature = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        
        # Following the feature extractor, replicate rest of the model layers described:
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.dense1 = tf.keras.layers.Dense(1024)
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(5)
        
    def call(self, inputs, training=False):
        # Forward pass through feature extractor
        x = self.feature(inputs, training=training)
        # Global average pooling reduces HxW dimension to vector
        x = self.global_avg_pool(x)
        # Dropout behaves differently in training vs inference mode
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instantiated MyModel object
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape of (batch, 64, 64, 3)
    # The batch size can be arbitrarily set to e.g. 4 for testing
    return tf.random.uniform((4, 64, 64, 3), dtype=tf.float32)

