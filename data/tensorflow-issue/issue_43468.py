# tf.random.uniform((128, 28, 28, 1), dtype=tf.float32) ← Batch size 128, 28x28 grayscale images, channels_last format

import tensorflow as tf
from tensorflow.keras import layers, utils

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture based on the MNIST ConvNet from the issue
        # Sequential layers unfolded into subclassed model
        
        # Note: input_shape (28,28,1) channels_last
        
        self.conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')
        self.maxpool = layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(10, activation='softmax')  # num_classes=10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return a fresh instance
    model = MyModel()
    # Compile model as in training code for full compatibility
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy'])
    return model

def GetInput():
    # Create a batch of 128 images (batch_size) with shape 28x28, grayscale (1 channel)
    return tf.random.uniform([128, 28, 28, 1], dtype=tf.float32)

