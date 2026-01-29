# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← CIFAR-100 images input shape (batch_size, 32, 32, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Simplified ResNet-like model wrapper.
    This reflects the usage of the ResNet56 model for CIFAR-100 from the issue.
    Since full ResNet implementation code is not provided,
    we provide a minimal mockup to illustrate integration and XLA usage.
    """
    def __init__(self, num_layers=56, num_class=100, name='ResNet', trainable=True):
        super().__init__(name=name)
        self.num_class = num_class
        self.trainable = trainable
        # Minimal simple model instead of full ResNet for demonstration
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2)
        # Suppose a few residual blocks replaced by a stack of conv layers as a placeholder
        self.conv_stack = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            self.pool,
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            self.pool,
            tf.keras.layers.Flatten(),
        ])
        self.fc = tf.keras.layers.Dense(num_class)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv_stack(x, training=training)
        logits = self.fc(x)
        return logits


def my_model_function():
    # Returns an instance of MyModel (ResNet-like model for CIFAR-100)
    return MyModel()


def GetInput():
    # Returns a random input tensor shaped as CIFAR-100 batch of RGB images
    # Assume batch size 128 as in the original code
    batch_size = 128
    height = 32
    width = 32
    channels = 3
    # dtype float32 as model expects float inputs normalized
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

