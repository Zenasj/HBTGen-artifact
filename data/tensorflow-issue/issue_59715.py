# tf.random.uniform((B, 256, 256, 32), dtype=tf.float32) ← inferred input shape from Input layer shape=(256, 256, 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the convolutional blocks and dense layers per original Sequential model
        self.conv_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        ])
        self.conv_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.relu = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.conv_block1(inputs, training=training)
        x = self.conv_block2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching input shape (batch_size, 256, 256, 32)
    # Here, batch size is assumed to be 1 as default for inference/testing.
    return tf.random.uniform((1, 256, 256, 32), dtype=tf.float32)

