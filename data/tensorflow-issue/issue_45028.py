# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) — inferred input shape and dtype from given datasets

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the simplest Conv2D model causing cudnn init error from the issue
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(10)  # num_classes=10, no activation to keep logits as in the issue

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x


def my_model_function():
    # Initialize the model instance
    model = MyModel()
    # Compile to match the original user code
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Matches input shape used in the issue: (batch_size=32, 28, 28, 1)
    # We'll create a batch dimension typical for training batch size 32
    # Data type float32 matching np.float32 dtype used
    
    batch_size = 32
    height = 28
    width = 28
    channels = 1
    # Create a random uniform tensor since original was normal, either fine for just input shape
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

