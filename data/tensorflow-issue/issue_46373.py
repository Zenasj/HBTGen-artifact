# tf.random.uniform((B, 300, 300, 3), dtype=tf.float32) ← inferred from image_size=(300,300) and typical 3-channel RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN model designed to match the example from the issue that runs on MirroredStrategy
        self.rescale = tf.keras.layers.Rescaling(1./255)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Output logits for 2 classes (per example)
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = self.rescale(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Build and compile the model similar to the example in the issue,
    # to run under a distribution strategy scope
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random images with shape (batch_size, 300, 300, 3)
    # Batch size 32 as per the example in the issue
    batch_size = 32
    height = 300
    width = 300
    channels = 3
    # Uniform random input between 0 and 1 as numeric example input
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

