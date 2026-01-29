# tf.random.uniform((B, 300, 300, 1), dtype=tf.float32)  # Input shape inferred from training pipeline

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple CNN architecture for digit classification on 300x300 grayscale images
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(300, 300, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Output layer: logits for 10 classes (digits 0-9)
        self.out = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.out(x)
        return logits

def my_model_function():
    # Instantiate the model and compile with optimizer, loss, and metrics
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def GetInput():
    # Generate a random batch of images simulating the input shape used for training
    # Batch size = 1 for simplicity, grayscale image with shape 300x300
    input_tensor = tf.random.uniform((1, 300, 300, 1), dtype=tf.float32)
    return input_tensor

